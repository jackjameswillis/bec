"""Unit tests for the BNNFitness batch schedules.

Checks:

1. ``schedule="cycle"`` walks the fit split as contiguous ``batch_size``
   batches in order (last batch partial), then wraps around;
2. ``schedule="random"`` still draws fixed-size batches without replacement
   per call, reproducibly under ``seed``;
3. an oversized batch is clamped to the dataset size in both modes;
4. unknown schedules raise.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes.fitness import BNNFitness


def constant(n, d=784, classes=10):
    x = np.zeros((n, d), np.float32)
    y = np.zeros((n, classes), np.float32)
    return tf.constant(x), tf.constant(y)


def fitness_fn(n, batch, schedule, seed=0):
    x, y = constant(n)
    return BNNFitness(x, y, batch_size=batch, seed=seed, schedule=schedule,
                      shapes=((784, 10),))


def main():
    n, batch = 10, 4

    # 1. cycle: contiguous slices in order, partial last batch, wraparound
    fn = fitness_fn(n, batch, "cycle")
    seqs = [fn._select_indices().numpy().tolist() for _ in range(5)]
    expected = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [0, 1, 2, 3], [4, 5, 6, 7]]
    assert seqs == expected, seqs

    # 2. random: fixed size without replacement, reproducible under seed
    fn_a, fn_b = fitness_fn(n, batch, "random", seed=3), fitness_fn(n, batch, "random", seed=3)
    draws_a = [fn_a._select_indices().numpy().tolist() for _ in range(4)]
    draws_b = [fn_b._select_indices().numpy().tolist() for _ in range(4)]
    assert draws_a == draws_b
    assert all(len(d) == batch and len(set(d)) == batch for d in draws_a)
    assert all(all(0 <= i < n for i in d) for d in draws_a)

    # 3. oversized batch clamps to n (random and cycle)
    for schedule in ("random", "cycle"):
        fn = fitness_fn(n, 128, schedule)
        fn.__call__(tf.ones((2, 7840), tf.bool), tf.zeros((2, 10)))
        assert fn.batch_size == n

    # 4. unknown schedule rejected
    try:
        fitness_fn(n, batch, "shuffle")._select_indices()
        raise AssertionError("unknown schedule accepted")
    except ValueError:
        pass

    # 5. invalid sizes rejected at construction
    x, y = constant(10)
    for kwargs in (dict(batch_size=0), dict(chunk_size=0), dict(pop_chunk=0)):
        try:
            BNNFitness(x, y, shapes=((784, 10),), **kwargs)
            raise AssertionError(f"accepted {kwargs}")
        except ValueError:
            pass

    print("test_fitness: all checks passed")


if __name__ == "__main__":
    main()
