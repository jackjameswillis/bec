"""Unit tests for the BES perturbation-table sampler.

Checks:

1. determinism: identical configuration yields identical populations, both
   in table and streaming mode;
2. circular wraparound: after consuming all T rows, sampling repeats the
   same individuals while theta is unchanged;
3. antithetic semantics: at p = 0.5 pair mates are exact complements, an odd
   population adds one unpaired row, antithetic table consumption uses
   ceil(P/2) rows per generation, and the Bernoulli rate at sigma(theta) =
   0.75 stays at 0.75 within Monte-Carlo error.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes.algs.es import BES


def populations(es, generations):
    return [es.sample().numpy() for _ in range(generations)]


def main():
    d, p, t = 8, 4, 16

    # 1. determinism (table mode and streaming mode)
    table = populations(BES(d, p, table_rows=t, seed=0), t // p)
    again = populations(BES(d, p, table_rows=t, seed=0), t // p)
    assert all(np.array_equal(a, b) for a, b in zip(table, again))
    stream = populations(BES(d, p, table_rows=None, seed=0), t // p)
    stream2 = populations(BES(d, p, table_rows=None, seed=0), t // p)
    assert all(np.array_equal(a, b) for a, b in zip(stream, stream2))

    # 2. wraparound after T rows with unchanged theta
    es = BES(d, p, table_rows=t, seed=0)
    first = es.sample().numpy()
    for _ in range(t // p - 1):
        es.sample()
    np.testing.assert_array_equal(first, es.sample().numpy())

    # 3a. antithetic at p = 0.5: second half == complement of first half
    es = BES(d, p, table_rows=t, antithetic=True, seed=0)
    pop = es.sample().numpy()
    k = p // 2
    np.testing.assert_array_equal(pop[k:2 * k], ~pop[:k])

    # 3b. odd population: extra unpaired tail row, mirror block intact
    es = BES(d, 5, table_rows=t, antithetic=True, seed=0)
    pop = es.sample().numpy()
    assert pop.shape == (5, d)
    np.testing.assert_array_equal(pop[2:4], ~pop[:2])
    assert not np.array_equal(pop[4], ~pop[0])

    # 3c. antithetic stays an exact Bernoulli(sigma(theta)) at p != 0.5
    # (tolerance ~4 SE: mirrored pairs raise pair-mean variance when the
    # integrand is asymmetric, var(pair mean) = 0.1875 at s = 0.75)
    theta = float(np.log(3.0))
    es = BES(d, 4096, table_rows=t, antithetic=True, seed=0)
    es.theta.assign(tf.fill([d], theta))
    rate = float(tf.reduce_mean(tf.cast(es.sample(), tf.float32)))
    assert abs(rate - 0.75) < 0.03, rate

    # antithetic half rows: 2 per generation, wraparound at same theta
    es = BES(d, p, table_rows=t, antithetic=True, seed=0)
    assert es.rows_per_generation() == 2
    before = es.sample().numpy()
    for _ in range(t // es.rows_per_generation() - 1):
        es.sample()
    np.testing.assert_array_equal(before, es.sample().numpy())

    # table_rows=0 disables the table (streaming); invalid inputs are rejected
    es = BES(d, p, table_rows=0, seed=0)
    assert es.table_rows is None
    assert es.sample().shape == (p, d)
    for kwargs in (dict(p_init=0.0), dict(table_rows=-1), dict(sigma=-1.0)):
        try:
            BES(d, p, **kwargs)
            raise AssertionError(f"accepted {kwargs}")
        except ValueError:
            pass
    try:
        BES(d, 0, seed=0)
        raise AssertionError("accepted population_size=0")
    except ValueError:
        pass

    print("perturbation-table sampler OK")


if __name__ == "__main__":
    main()
