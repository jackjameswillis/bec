"""Fitness-shaping test for the soft Bernoulli ES.

Checks BES._utility against hand-computed utilities for the three
shaping modes ("none", "standardize", "rank"), including stable tie-breaking
and the all-equal-fitness degenerate case, plus an end-to-end update() parity
check for each mode on a constructed population.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes import mlp
from bes.algs._common import fitness_utility
from bes.algs.es import BES

NUM_BITS = 8
POP = 6


def reference_rank(f):
    f = np.asarray(f, dtype=np.float64)
    p = f.shape[0]
    if p <= 1:
        return np.zeros_like(f)
    ranks = np.empty(p, dtype=np.float64)
    ranks[np.argsort(f, kind="stable")] = np.arange(p)
    return ranks / (p - 1.0) - 0.5


def build_es(shaping, fitness, num_bias=0, sigma=0.25, seed=0):
    es = BES(NUM_BITS, POP, p_init=0.5, seed=seed, num_bias=num_bias,
             sigma=sigma, shaping=shaping)
    es.population = tf.zeros((POP, NUM_BITS), dtype=tf.bool)
    es.fitness = tf.convert_to_tensor(np.asarray(fitness, np.float32))
    return es


def main():
    rng = np.random.default_rng(5)
    fitness = rng.normal(size=POP)

    # --- "none": raw advantage
    u = build_es("none", fitness)._utility().numpy()
    np.testing.assert_allclose(u, fitness - fitness.mean(), rtol=1e-6, atol=1e-7)

    # --- "standardize": population z-score; all-equal fitness gives zeros
    u = build_es("standardize", fitness)._utility().numpy()
    np.testing.assert_allclose(u.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(u.std(), 1.0, rtol=1e-5)
    u = build_es("standardize", np.full(POP, 2.5))._utility().numpy()
    np.testing.assert_allclose(u, 0.0, atol=1e-7)

    # --- "rank": centered ordinal ranks; hand-computed cases
    # distinct values -> u in [-0.5, 0.5], sum 0
    u = build_es("rank", fitness)._utility().numpy()
    assert u.min() >= -0.5 - 1e-6 and u.max() <= 0.5 + 1e-6, (u.min(), u.max())
    np.testing.assert_allclose(u.sum(), 0.0, atol=1e-6)
    order = np.argsort(np.argsort(fitness)) + 1  # ordinal ranks 1..P
    np.testing.assert_allclose(
        u, (order - (POP + 1) / 2.0) / (POP - 1), rtol=1e-6, atol=1e-7
    )

    # ties use ordinal ranks, broken by index: [1, 2, 2, 3, 3, 9] -> ranks 0..5
    u = build_es("rank", [1.0, 2.0, 2.0, 3.0, 3.0, 9.0])._utility().numpy()
    np.testing.assert_allclose(
        u, np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]), rtol=1e-6, atol=1e-7
    )

    # all-equal fitness -> fixed index-order spread (no tie averaging)
    u = build_es("rank", np.full(POP, -1.0))._utility().numpy()
    np.testing.assert_allclose(
        u, np.arange(POP) / (POP - 1.0) - 0.5, rtol=1e-6, atol=1e-7)

    # vectorised argsort rank matches the ordinal-rank reference across sizes/ties
    for p in (1, 2, 3, 6, 17, 64, 257):
        for values in (rng.normal(size=p),
                       rng.integers(0, 4, size=p).astype(np.float64)):
            got = fitness_utility(
                tf.constant(values, dtype=tf.float32), "rank").numpy()
            np.testing.assert_allclose(
                got, reference_rank(values), rtol=1e-6, atol=1e-7)

    # large population: bounded centred ordinal ranks, no (P, P) allocation
    big = rng.normal(size=4096).astype(np.float32)
    u = fitness_utility(tf.constant(big), "rank").numpy()
    assert u.min() >= -0.5 - 1e-6 and u.max() <= 0.5 + 1e-6
    np.testing.assert_allclose(u.sum(), 0.0, atol=2e-4)

    # --- update() parity for each mode (weights + biases)
    num_bias = mlp.num_bias(2, 2, 2)
    s = np.full(NUM_BITS, 0.5)  # p_init 0.5 -> s = 0.5
    bias_draw = rng.normal(size=(POP, num_bias)).astype("float32")
    lr = 0.05
    for shaping, ref in [
        ("none", fitness - fitness.mean()),
        ("standardize", (fitness - fitness.mean()) / fitness.std()),
        ("rank", (np.argsort(np.argsort(fitness)) + 1 - (POP + 1) / 2.0) / (POP - 1)),
    ]:
        es = build_es(shaping, fitness, num_bias=num_bias, sigma=0.25)
        es.population = tf.convert_to_tensor(
            rng.integers(0, 2, size=(POP, NUM_BITS)).astype(bool))
        es.bias = tf.convert_to_tensor(bias_draw)
        before_w, before_b = es.theta.numpy().copy(), es.theta_bias.numpy().copy()
        es.update(lr)
        u_ref = np.asarray(ref, np.float64)
        delta_w_ref = lr * (u_ref[:, None] * (
            es.population.numpy().astype(np.float64) - s)).sum(axis=0)
        delta_b_ref = lr * (u_ref[:, None] * (
            bias_draw - 0.0) / 0.25 ** 2).sum(axis=0)
        np.testing.assert_allclose(es.theta.numpy() - before_w, delta_w_ref,
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(es.theta_bias.numpy() - before_b, delta_b_ref,
                                   rtol=1e-5, atol=1e-6)

    # --- bias_lr decouples the bias step from the weight lr (score unchanged)
    es = BES(NUM_BITS, POP, p_init=0.5, seed=0, num_bias=num_bias,
             sigma=0.25, shaping="none", bias_lr=0.5)
    es.population = tf.convert_to_tensor(
        rng.integers(0, 2, size=(POP, NUM_BITS)).astype(bool))
    es.bias = tf.convert_to_tensor(bias_draw)
    es.fitness = tf.convert_to_tensor(np.asarray(fitness, np.float32))
    before_w, before_b = es.theta.numpy().copy(), es.theta_bias.numpy().copy()
    es.update(0.05)
    u_ref = np.asarray(fitness - fitness.mean(), np.float64)
    delta_b_ref = 0.5 * (u_ref[:, None] * (bias_draw - 0.0) / 0.25 ** 2).sum(axis=0)
    np.testing.assert_allclose(es.theta_bias.numpy() - before_b, delta_b_ref,
                               rtol=1e-5, atol=1e-6)
    delta_w_ref = 0.05 * (u_ref[:, None] * (
        es.population.numpy().astype(np.float64) - 0.5)).sum(axis=0)
    np.testing.assert_allclose(es.theta.numpy() - before_w, delta_w_ref,
                               rtol=1e-5, atol=1e-6)

    # --- adam: two updates parity against a manual numpy Adam reference
    es = BES(NUM_BITS, POP, p_init=0.5, seed=0, sigma=0.0,
             shaping="none", adam=True)
    es.population = tf.convert_to_tensor(
        rng.integers(0, 2, size=(POP, NUM_BITS)).astype(bool))
    es.fitness = tf.convert_to_tensor(np.asarray(fitness, np.float32))
    lr, b1, b2, eps = 0.02, 0.9, 0.999, 1e-8
    theta_ref = np.zeros(NUM_BITS)
    m = np.zeros(NUM_BITS)
    v = np.zeros(NUM_BITS)
    for t in (1, 2):
        es.update(lr)
        s_ref = 1.0 / (1.0 + np.exp(-theta_ref))
        u_ref = np.asarray(fitness - fitness.mean(), np.float64)
        g = (u_ref[:, None] * (
            es.population.numpy().astype(np.float64) - s_ref)).sum(axis=0)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g ** 2
        theta_ref += lr * (m / (1.0 - b1 ** t)) / (
            np.sqrt(v / (1.0 - b2 ** t)) + eps)
        np.testing.assert_allclose(es.theta.numpy(), theta_ref,
                                   rtol=1e-5, atol=1e-6)

    # --- grad_chunk: chunked population reduction == full shot (fp order only)
    es = BES(NUM_BITS, POP, p_init=0.5, seed=0, sigma=0.0, shaping="none")
    es.population = tf.convert_to_tensor(
        rng.integers(0, 2, size=(POP, NUM_BITS)).astype(bool))
    es.fitness = tf.convert_to_tensor(np.asarray(fitness, np.float32))
    es.update(0.05)
    full = es.theta.numpy().copy()
    es2 = BES(NUM_BITS, POP, p_init=0.5, seed=0, sigma=0.0, shaping="none")
    es2.population = es.population
    es2.fitness = es.fitness
    es2.update(0.05, grad_chunk=2)
    np.testing.assert_allclose(es2.theta.numpy(), full, rtol=1e-5, atol=1e-6)

    # --- pop_chunk: population-axis chunking of fitness eval is row-exact
    rng2 = np.random.default_rng(7)
    xr = tf.constant(rng2.normal(size=(200, mlp.INPUT_DIM)).astype("float32"))
    yr = tf.one_hot(tf.range(200) % mlp.NUM_CLASSES, mlp.NUM_CLASSES)
    bits_r = tf.convert_to_tensor(rng2.integers(0, 2, size=(5, mlp.num_bits())).astype(bool))
    loss_full, acc_full = mlp.population_metrics(bits_r, xr, yr, chunk_size=100)
    loss_ch, acc_ch = mlp.population_metrics(bits_r, xr, yr, chunk_size=100, pop_chunk=2)
    np.testing.assert_array_equal(loss_ch.numpy(), loss_full.numpy())
    np.testing.assert_array_equal(acc_ch.numpy(), acc_full.numpy())

    # --- natural: p_init 0.5 -> s(1-s) = 0.25 -> grad scaled by 4 (eps ~0)
    es = BES(NUM_BITS, POP, p_init=0.5, seed=0, sigma=0.0,
             shaping="none", natural=True, natural_eps=1e-8)
    es.population = tf.convert_to_tensor(
        rng.integers(0, 2, size=(POP, NUM_BITS)).astype(bool))
    es.fitness = tf.convert_to_tensor(np.asarray(fitness, np.float32))
    es.update(0.05)
    u_ref = np.asarray(fitness - fitness.mean(), np.float64)
    nat_ref = 0.05 * (u_ref[:, None] * (
        es.population.numpy().astype(np.float64) - 0.5)).sum(axis=0) / 0.25
    np.testing.assert_allclose(es.theta.numpy(), nat_ref, rtol=1e-5, atol=1e-6)

    # unknown shaping is rejected
    try:
        build_es("bogus", fitness)._utility()
        raise AssertionError("should have raised")
    except ValueError as e:
        print("unknown shaping raises OK:", e)

    print("fitness shaping OK: none / standardize / rank")


if __name__ == "__main__":
    main()
