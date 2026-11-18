"""Unit tests for the MicrobialGA ring pairing and tournament operators.

Checks:

1. pairing: perfect matching (every index appears exactly once, no self
   pairs), partner containment inside each other's ring deme window, the
   deme=2 adjacent special case, determinism under a fixed seed, and the
   odd-population / tiny-deme value errors;
2. tournament semantics: winners keep their exact rows, losers get the
   winner's genome under crossrate=1, keep their genome under crossrate=0,
   and get every bit flipped under mutation_rate=1; winner rows never change;
3. seeded runs reproduce identical populations (and different seeds differ);
4. toy selection: on deterministic count-of-ones fitness the population best
   never decreases and reaches the optimum within 300 generations.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes.algs.mga import MicrobialGA


def check_pairing(pairs, pop, deme):
    assert pairs.shape == (pop // 2, 2)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    seen = set()
    for a, b in pairs:
        assert a != b
        for idx in (a, b):
            assert idx not in seen, f"index {idx} paired twice"
            seen.add(idx)
    assert len(seen) == pop
    if deme == 2:
        for a, b in pairs:
            assert (b - a) % pop == 1 or (a - b) % pop == 1
    else:
        d = deme if deme < pop else pop
        for a, b in pairs:
            assert ((b - a) % pop <= d - 1) or ((a - b) % pop <= d - 1), (a, b)


def main():
    pop, d_bits, d_bias = 8, 16, 2

    # 1. pairing validity across deme sizes
    for deme in (2, 3, 5, pop):
        mga = MicrobialGA(d_bits, pop, deme_size=deme, seed=0)
        check_pairing(mga.pair(), pop, deme)
    # deme=2 partner is an exact ring neighbour for every generation
    mga = MicrobialGA(d_bits, pop, deme_size=2, seed=1)
    for _ in range(5):
        check_pairing(mga.pair(), pop, 2)
    # determinism / seed separation
    ga_a = MicrobialGA(d_bits, pop, deme_size=3, seed=5)
    ga_b = MicrobialGA(d_bits, pop, deme_size=3, seed=5)
    ga_c = MicrobialGA(d_bits, pop, deme_size=3, seed=6)
    pairs_a, pairs_b, pairs_c = [], [], []
    for _ in range(3):
        pairs_a.append(ga_a.pair())
        pairs_b.append(ga_b.pair())
        pairs_c.append(ga_c.pair())
    assert all(np.array_equal(x, y) for x, y in zip(pairs_a, pairs_b))
    assert not all(np.array_equal(x, y) for x, y in zip(pairs_a, pairs_c))
    # value errors
    try:
        MicrobialGA(d_bits, pop + 1, seed=0)
        raise AssertionError("odd population accepted")
    except ValueError:
        pass
    try:
        MicrobialGA(d_bits, pop, deme_size=1, seed=0)
        raise AssertionError("deme_size < 2 accepted")
    except ValueError:
        pass

    # 2. tournament semantics with hand-set fitness
    fitness = tf.constant([1.0, 3.0, 2.0, 0.0])

    def winners_of(mga):
        res = []
        for a, b in mga.pairs:
            res.append(a if mga.fitness.numpy()[a] >= mga.fitness.numpy()[b] else b)
        return np.array(res, dtype=int)

    def run_tournament(crossrate, mutation_rate, bias_std, deme=pop):
        mga = MicrobialGA(d_bits, len(fitness), num_bias=d_bias, deme_size=deme,
                          crossrate=crossrate, mutation_rate=mutation_rate,
                          bias_std=bias_std, seed=11)
        mga.pair()
        mga.fitness = fitness
        before_bits = mga.population.numpy().copy()
        before_bias = mga.bias.numpy().copy()
        mga.tournaments()
        return mga, before_bits, before_bias

    # 2a. crossrate=1, no mutation: loser becomes the winner's genome
    mga, before_bits, before_bias = run_tournament(1.0, 0.0, 0.0)
    winners = winners_of(mga)
    losers = np.array(list(set(range(len(fitness))) - set(winners)), dtype=int)
    after_bits = mga.population.numpy()
    after_bias = mga.bias.numpy()
    np.testing.assert_array_equal(after_bits[winners], before_bits[winners])
    np.testing.assert_array_equal(after_bias[winners], before_bias[winners])
    for w, l in zip(winners, losers):
        np.testing.assert_array_equal(after_bits[l], before_bits[w])
        np.testing.assert_array_equal(after_bias[l], before_bias[w])
    assert np.array_equal(mga.best_bits().numpy(), before_bits[winners[1]])

    # 2b. crossrate=0, no mutation: genomes unchanged, loser biases get noise
    mga, before_bits, before_bias = run_tournament(0.0, 0.0, 1.0)
    winners = winners_of(mga)
    losers = np.array(list(set(range(len(fitness))) - set(winners)), dtype=int)
    after_bits = mga.population.numpy()
    after_bias = mga.bias.numpy()
    np.testing.assert_array_equal(after_bits[winners], before_bits[winners])
    np.testing.assert_array_equal(after_bits[losers], before_bits[losers])
    np.testing.assert_array_equal(after_bias[winners], before_bias[winners])
    assert np.any(after_bias[losers] != before_bias[losers])
    assert np.all(np.isfinite(after_bias))

    # 2c. mutation_rate=1 flips every loser bit; winners untouched
    mga, before_bits, before_bias = run_tournament(0.0, 1.0, 0.0)
    winners = winners_of(mga)
    losers = np.array(list(set(range(len(fitness))) - set(winners)), dtype=int)
    after_bits = mga.population.numpy()
    np.testing.assert_array_equal(after_bits[winners], before_bits[winners])
    np.testing.assert_array_equal(after_bits[losers], ~before_bits[losers])
    np.testing.assert_array_equal(mga.bias.numpy(), before_bias)

    # 3. seeded runs are reproducible; different seeds diverge
    def count_ones(bits, bias):
        return tf.reduce_sum(tf.cast(bits, tf.float32), axis=1)

    def seeded_population(seed):
        mga = MicrobialGA(d_bits, pop, num_bias=d_bias, deme_size=pop,
                          crossrate=0.5, mutation_rate=0.1, bias_std=0.1,
                          seed=seed)
        states = []
        for _ in range(4):
            mga.step(count_ones)
            states.append((mga.population.numpy(), mga.bias.numpy()))
        return states

    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
               for x, y in zip(seeded_population(4), seeded_population(4)))
    assert any(not np.array_equal(x[0], y[0])
               for x, y in zip(seeded_population(4), seeded_population(5)))

    # 4. toy convergence: deterministic fitness, best fitness never decreases
    mga = MicrobialGA(64, 16, deme_size=16, crossrate=0.75, mutation_rate=0.01,
                      seed=7)
    trace = []
    for _ in range(300):
        mga.step(count_ones)
        trace.append(int(mga.fitness_best()))
    assert all(b >= a for a, b in zip(trace, trace[1:])), trace
    assert trace[-1] == 64, trace[-1]
    assert trace[0] < 64, trace[0]

    # bias-free individuals step without error
    mga = MicrobialGA(16, 4, num_bias=0, seed=0)
    mga.step(count_ones)
    assert mga.population.shape == (4, 16)
    assert mga.best_bias().shape == (0,)

    print("test_mga: all checks passed")


if __name__ == "__main__":
    main()
