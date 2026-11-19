"""Microbial genetic algorithm (Harvey-style) over persistent binary individuals.

Population-based greedy replacement genetic algorithm, ported from the
PyTorch reference ``/data/mgab`` (``PopMLP.tournaments`` + ``geography.Ring``)
onto the shared TF fitness layer. Unlike ``bes/algs/es.py`` the population is
persistent (uniform Bernoulli 0.5 weight-bit init, zero biases) and no
distribution parameters are learned; fitness (maximised, comb-exp convention)
drives pairwise replacement.

Every iteration:

1. **Pairing** — a random perfect matching under ring-deme constraints
   (``geography.Ring.tournament`` port): starting from a random ring position,
   each still-unpaired individual picks a *random unpaired* partner within its
   forward window ``{i+1, ..., i+deme_size-1} (mod P)``. ``deme_size >= P``
   degenerates to a fully random matching; ``deme_size == 2`` is equivalent to
   the adjacent-pair matching that greedy produces (both special-cased,
   vectorised). The population must be even (perfect matching).
2. **Evaluation** — ``fitness_fn(population, bias)`` scores the whole
   population (bulk BNN fitness in ``bes/fitness.py``: one shared random batch
   per call, the mgab ``score_mode='batch'`` protocol).
3. **Winner/loser** — per pair the higher fitness wins (``>=``; tie-break: the
   lower pair index wins rather than replicating mgab's random ring-order
   tie-break — an exact-fitness tie is measure-zero for continuous fitness).
4. **Crossover** — per weight bit (and per bias coordinate), the loser copies
   the winner's gene independently with probability ``crossrate`` (uniform
   crossover); ``crossrate = 1`` clones the winner, ``0`` leaves the loser.
5. **Mutation (losers only)** — weight bits flip independently with
   probability ``mutation_rate``; continuous biases get Gaussian noise
   ``N(0, bias_std)`` on top of the crossover. Winners are untouched, so the
   population's best individual is never destroyed (only losers are
   overwritten in place).
"""

import numpy as np
import tensorflow as tf

from bes.algs._common import row_chunk_size


class MicrobialGA:
    """Persistent population of binary individuals evolved by tournaments.

    Weight bits are flat ``{0,1}^D`` (continuous layer biases ``(P, D_b)``
    ride along per individual, zero-initialised); the pairing, crossover and
    mutation operators follow the mgab microbial GA. All tensor ops run on
    a seeded ``tf.random.Generator``; the pairing index bookkeeping uses a
    seeded numpy generator.
    """

    def __init__(self, num_bits, population_size, num_bias=0, deme_size=2,
                 crossrate=0.5, mutation_rate=1e-3, bias_std=0.01, seed=None):
        if int(num_bits) <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")
        if population_size % 2:
            raise ValueError(f"population_size must be even, got {population_size}")
        deme = int(deme_size)
        if deme < 2:
            raise ValueError(f"deme_size must be >= 2, got {deme_size}")
        if not 0.0 <= float(crossrate) <= 1.0:
            raise ValueError(f"crossrate must be in [0, 1], got {crossrate}")
        if float(mutation_rate) < 0.0:
            raise ValueError(f"mutation_rate must be non-negative, got {mutation_rate}")
        if float(bias_std) < 0.0:
            raise ValueError(f"bias_std must be non-negative, got {bias_std}")
        self.num_bits = int(num_bits)
        self.num_bias = int(num_bias)
        self.population_size = int(population_size)
        self.deme_size = deme
        self.crossrate = float(crossrate)
        self.mutation_rate = float(mutation_rate)
        self.bias_std = float(bias_std)
        seed = 0 if seed is None else seed
        self._gen = tf.random.Generator.from_seed(seed)
        self._pair_rng = np.random.default_rng(seed)
        # adaptive row-chunk: bounds (rows, D) float noise temporaries to
        # ~1 GB for wide nets (same rule as BES streaming sampling)
        self._chunk = row_chunk_size(self.num_bits)
        self.population = tf.concat(
            [self._gen.uniform([min(self._chunk, self.population_size - i), self.num_bits],
                               0.0, 1.0) < 0.5
             for i in range(0, self.population_size, self._chunk)],
            axis=0,
        )
        self.bias = tf.zeros([self.population_size, self.num_bias], tf.float32)
        self.fitness = None
        self.pairs = None

    def _bool_mask(self, rows, threshold):
        """(rows, D) bool noise mask built in row chunks (RNG order preserved)."""
        return tf.concat(
            [self._gen.uniform([min(self._chunk, rows - i), self.num_bits],
                               0.0, 1.0) < threshold
             for i in range(0, rows, self._chunk)],
            axis=0,
        )

    def pair(self):
        """Random perfect matching under ring-deme constraints: (P//2, 2).

        Unordered pairs as numpy int64, assignments symmetric; stored for
        introspection. Reproduces ``geography.Ring.tournament`` outcomes.
        """
        pop = self.population_size
        selected = np.full(pop, -1, dtype=np.int64)
        if self.deme_size >= pop:
            perm = self._pair_rng.permutation(pop)
            selected[perm[0::2]] = perm[1::2]
            selected[perm[1::2]] = perm[0::2]
        else:
            window = self.deme_size - 1
            start = int(self._pair_rng.integers(pop))
            if window == 1:
                # deme_size == 2: the greedy walk reduces to adjacent pairs
                # rotated by the random start.
                sequence = (start + np.arange(pop)) % pop
                selected[sequence[0::2]] = sequence[1::2]
                selected[sequence[1::2]] = sequence[0::2]
            else:
                for i in range(pop):
                    idx = (start + i) % pop
                    if selected[idx] != -1:
                        continue
                    free = [(idx + w) % pop for w in range(1, window + 1)
                            if selected[(idx + w) % pop] == -1]
                    if free:
                        chosen = free[int(self._pair_rng.integers(len(free)))]
                        selected[idx] = chosen
                        selected[chosen] = idx
        mask = np.arange(pop) < selected
        self.pairs = np.stack(
            [np.arange(pop)[mask], selected[mask]], axis=1
        )
        return self.pairs

    def evaluate(self, fitness_fn):
        self.fitness = fitness_fn(self.population, self.bias)

    def tournaments(self):
        """Apply winner/loser crossover + mutation for the last ``pair()``/``evaluate()``.

        Losers' weight bits are replaced by winner genes with probability
        ``crossrate``, then flipped with probability ``mutation_rate``; loser
        biases cross over per coordinate then get ``N(0, bias_std)`` noise.
        Winners (and all rows not in a losing slot) are left unchanged.
        """
        pairs = tf.constant(self.pairs, tf.int32)
        a, b = pairs[:, 0], pairs[:, 1]
        a_wins = tf.gather(self.fitness, a) >= tf.gather(self.fitness, b)
        winners = tf.where(a_wins, a, b)
        losers = tf.where(a_wins, b, a)
        count = self.population_size // 2
        win_bits = tf.gather(self.population, winners)
        lose_bits = tf.gather(self.population, losers)
        if self.crossrate < 1.0:
            mask = self._bool_mask(count, self.crossrate)
            lose_bits = tf.where(mask, win_bits, lose_bits)
        else:
            lose_bits = win_bits
        if self.mutation_rate > 0.0:
            flips = self._bool_mask(count, self.mutation_rate)
            lose_bits = tf.math.logical_xor(lose_bits, flips)
        self.population = tf.tensor_scatter_nd_update(
            self.population, losers[:, None], lose_bits
        )
        if self.num_bias > 0:
            win_bias = tf.gather(self.bias, winners)
            lose_bias = tf.gather(self.bias, losers)
            if self.crossrate < 1.0:
                mask = self._gen.uniform([count, self.num_bias], 0.0, 1.0) < self.crossrate
                lose_bias = tf.where(mask, win_bias, lose_bias)
            else:
                lose_bias = win_bias
            if self.bias_std > 0.0:
                noise = self._gen.normal(
                    [count, self.num_bias], stddev=self.bias_std
                )
                lose_bias = lose_bias + noise
            self.bias = tf.tensor_scatter_nd_update(
                self.bias, losers[:, None], lose_bias
            )

    def step(self, fitness_fn):
        self.pair()
        self.evaluate(fitness_fn)
        self.tournaments()

    def fitness_mean(self):
        return tf.reduce_mean(self.fitness)

    def fitness_best(self):
        return tf.reduce_max(self.fitness)

    def best_index(self):
        return tf.argmax(self.fitness)

    def best_bits(self):
        """Highest-fitness individual: (D,) bool."""
        return tf.gather(self.population, self.best_index())

    def best_bias(self):
        """Bias vector of the highest-fitness individual: (D_b,) float32."""
        return tf.gather(self.bias, self.best_index())
