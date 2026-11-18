"""Unit tests for the Gaussian (sign-quantized) ES.

Checks:

1. determinism: identical configuration yields identical populations, both
   in table and streaming mode;
2. known-noise parity: with an injected standard-normal table the population
   and the REINFORCE update match a hand-rolled numpy reference exactly,
   across several generations with circular table wraparound (antithetic
   pairing, odd-population tail, mirrored score -eps);
3. streaming re-draw consistency: the update replays the same noise stream
   the sample consumed (rebuilt independently in the test from the
   per-generation mixed seed);
4. REINFORCE score unbiasedness for a linear fitness against the closed-form
   gradient of J(theta) = sum_j a_j (2 Phi(theta_j / sigma) - 1);
5. shaping utilities identical to SoftES on the same fitness vector;
6. grad_chunk parity (fp summation order only, noise stream unchanged);
7. diagnostics at theta = 0: entropy 1 bit, flip probability 0.5, committed
   fraction 0.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes.algs.es import SoftES
from bes.algs.ges import GaussianES, _noise_seed


def inject_table(es, table, theta_vals):
    """Replace the generated noise table / theta with known values."""
    es._table = tf.constant(table)
    es._rows_used = 0
    es.theta.assign(tf.constant(theta_vals, tf.float32))
    return es


def numpy_reference(theta, sigma, table, start, pair_rows, extra,
                    antithetic, fitness, lr):
    """Hand-rolled population + REINFORCE update for one generation.

    Mirrors the chunk layout of GaussianES: paired chunks contribute
    ``u_plus @ eps + u_minus @ (-eps)`` with the ``+eps`` half first.
    """
    t = table.shape[0]
    idx = np.arange(start, start + pair_rows + extra) % t
    rows = table[idx]
    k = pair_rows
    e = sigma * rows[:k]
    parts = [(theta[None] + e) > 0.0, (theta[None] - e) > 0.0]
    u = fitness - fitness.mean()
    grad = u[:k] @ rows[:k]
    if antithetic:
        grad = grad + u[k:2 * k] @ (-rows[:k])
        if extra:
            parts.append((theta[None] + sigma * rows[k:k + extra]) > 0.0)
            grad = grad + u[2 * k:2 * k + extra] @ rows[k:k + extra]
    pop = np.concatenate(parts, axis=0)
    return pop, theta + np.float32(lr) * grad / sigma


def main():
    d, t = 8, 16

    # 1. determinism (table mode and streaming mode)
    def pops(es, n):
        return [es.sample().numpy() for _ in range(n)]

    table_runs = pops(GaussianES(d, 4, sigma=0.7, table_rows=t, seed=0), 5)
    again = pops(GaussianES(d, 4, sigma=0.7, table_rows=t, seed=0), 5)
    assert all(np.array_equal(a, b) for a, b in zip(table_runs, again))
    stream = pops(GaussianES(d, 4, sigma=0.7, table_rows=None, seed=0), 5)
    stream2 = pops(GaussianES(d, 4, sigma=0.7, table_rows=None, seed=0), 5)
    assert all(np.array_equal(a, b) for a, b in zip(stream, stream2))

    # 1b. antithetic: population rows are sign(theta +/- sigma*eps) pairs,
    #     odd population adds one unpaired tail row
    es = GaussianES(d, 5, sigma=0.7, table_rows=t, antithetic=True, seed=0)
    pop = es.sample().numpy()
    assert pop.shape == (5, d)
    theta = es.theta.numpy()
    np.testing.assert_array_equal(pop[:2], (theta + 0.7 * es._table.numpy()[:2]) > 0)
    np.testing.assert_array_equal(pop[2:4], (theta - 0.7 * es._table.numpy()[:2]) > 0)

    # 2. known-noise parity over 4 generations with wraparound
    rng = np.random.default_rng(1)
    table = rng.standard_normal((t, d)).astype(np.float32)
    theta0 = rng.standard_normal(d).astype(np.float32)
    lr, sigma = 0.5, 0.7
    es = inject_table(GaussianES(d, 10, sigma=sigma, table_rows=t,
                                 antithetic=True, seed=0), table, theta0)
    ref_theta = theta0.copy()
    rows_used = 0
    for gen in range(4):
        es.sample()
        pop = es.population.numpy()
        fitness = pop.astype(np.float32) @ rng.standard_normal(d).astype(np.float32)
        es.fitness = tf.constant(fitness)
        es.update(lr)
        ref_pop, ref_theta = numpy_reference(
            ref_theta, sigma, table, rows_used, 10 // 2, 10 % 2, True,
            fitness, lr)
        rows_used += es.rows_per_generation()
        np.testing.assert_array_equal(pop, ref_pop)
        np.testing.assert_allclose(es.theta.numpy(), ref_theta, rtol=1e-5, atol=1e-6)

    # 2b. non-antithetic single-chunk parity (odd population, no extra row)
    es = inject_table(GaussianES(d, 7, sigma=sigma, table_rows=t, seed=0),
                      table, theta0)
    es.sample()
    fitness = rng.standard_normal(7).astype(np.float32)
    es.fitness = tf.constant(fitness)
    es.update(2.0 * lr)
    _, ref_theta = numpy_reference(theta0, sigma, table, 0, 7, 0, False,
                                   fitness, 2.0 * lr)
    np.testing.assert_allclose(es.theta.numpy(), ref_theta, rtol=1e-5, atol=1e-6)

    # 3. streaming re-draw consistency: rebuild the noise independently
    d3, p3, sigma3, seed3 = 8, 5, 0.7, 3
    es = GaussianES(d3, p3, sigma=sigma3, table_rows=None, antithetic=True,
                    seed=seed3)
    theta = es.theta.numpy()
    es.sample()
    gen = tf.random.Generator.from_seed(_noise_seed(seed3, 1))
    eps_pair = gen.normal([p3 // 2, d3], stddev=1.0).numpy()
    eps_extra = gen.normal([1, d3], stddev=1.0).numpy()
    ref_pop = np.concatenate([
        (theta + sigma3 * eps_pair) > 0.0,
        (theta - sigma3 * eps_pair) > 0.0,
        (theta + sigma3 * eps_extra) > 0.0,
    ])
    np.testing.assert_array_equal(es.population.numpy(), ref_pop)
    fitness = ref_pop.astype(np.float32) @ rng.standard_normal(d3).astype(np.float32)
    es.fitness = tf.constant(fitness)
    es.update(0.5)
    u = fitness - fitness.mean()
    grad = (u[:2] @ eps_pair + u[2:4] @ (-eps_pair) + u[4:5] @ eps_extra) / sigma3
    np.testing.assert_allclose(
        es.theta.numpy(), theta + np.float32(0.5) * grad, rtol=1e-5, atol=1e-6)

    # 4. score unbiasedness on a linear fitness: E[f * eps_j / sigma]
    #    == d/dtheta_j sum_i a_i (2 Phi(theta_i / sigma) - 1) = a_j 2 phi(theta_j/sigma) / sigma
    d4, p4, sigma4 = 3, 200_000, 1.0
    a = np.array([1.0, -2.0, 0.5], np.float32)
    theta4 = np.array([0.5, -1.0, 2.0], np.float32)
    es = inject_table(GaussianES(d4, p4, sigma=sigma4, table_rows=p4, seed=0),
                      rng.standard_normal((p4, d4)).astype(np.float32), theta4)
    es.sample()

    def fitness_fn(bits, bias):
        z = 2.0 * tf.cast(bits, tf.float32) - 1.0
        return tf.reduce_sum(z * a, axis=1)

    es.evaluate(fitness_fn)
    f = es.fitness.numpy()
    est = (f[:, None] * es._table.numpy()).mean(axis=0) / sigma4
    phi = np.exp(-0.5 * (theta4 / sigma4) ** 2) / np.sqrt(2.0 * np.pi)
    exact = a * 2.0 * phi / sigma4
    np.testing.assert_allclose(est, exact, rtol=0.05, atol=0.02)

    # 5. shaping utilities identical to SoftES
    f5 = rng.normal(size=17).astype(np.float32)
    for shaping in ("none", "standardize", "rank"):
        ges = GaussianES(4, 17, shaping=shaping, table_rows=None)
        soft = SoftES(4, 17, shaping=shaping, table_rows=None)
        ges.fitness = tf.constant(f5)
        soft.fitness = tf.constant(f5)
        np.testing.assert_allclose(
            ges._utility().numpy(), soft._utility().numpy(), rtol=1e-6, atol=1e-6)
    ges = GaussianES(4, 17, shaping="rank", table_rows=None)
    ges.fitness = tf.ones([17])
    np.testing.assert_allclose(ges._utility().numpy(), np.zeros(17), atol=1e-6)

    # 6. grad_chunk parity (fp summation order only)
    table6 = rng.standard_normal((64, 32)).astype(np.float32)
    theta6 = rng.standard_normal(32).astype(np.float32)
    es_a = inject_table(GaussianES(32, 20, sigma=0.7, table_rows=64,
                                   antithetic=True, seed=0), table6, theta6)
    es_b = inject_table(GaussianES(32, 20, sigma=0.7, table_rows=64,
                                   antithetic=True, seed=0), table6, theta6)
    f6 = rng.standard_normal(20).astype(np.float32)
    es_a.sample()
    es_b.sample()
    es_a.fitness = tf.constant(f6)
    es_b.fitness = tf.constant(f6)
    es_a.update(0.3)
    es_b.update(0.3, grad_chunk=3)
    np.testing.assert_allclose(es_a.theta.numpy(), es_b.theta.numpy(),
                               rtol=1e-5, atol=1e-6)

    # 7. diagnostics at theta = 0
    es = GaussianES(8, 4, sigma=1.0, table_rows=None, seed=0)
    es.theta.assign(tf.zeros([8]))
    diag = es.diagnostics()
    assert abs(diag["entropy_bits"] - 1.0) < 1e-6
    assert diag["committed_frac"] == 0.0
    assert abs(diag["flip_frac"] - 0.5) < 1e-6

    # 8. table_rows=0 disables the table; invalid inputs are rejected
    es = GaussianES(8, 4, sigma=1.0, table_rows=0, seed=0)
    assert es.table_rows is None
    assert es.sample().shape == (4, 8)
    for kwargs in (dict(sigma=0.0), dict(init_std=0.0), dict(table_rows=-1)):
        try:
            GaussianES(8, 4, **kwargs)
            raise AssertionError(f"accepted {kwargs}")
        except ValueError:
            pass

    print("test_ges: all checks passed")


if __name__ == "__main__":
    main()
