"""Fitness for populations of binary nets.

Convention (as in comb-exp): algorithms maximise fitness, so the
minimisation problem (training loss) is negated here before it is passed to
the ES. ``kind="loss"`` returns ``-loss`` (default loss kind ``"ce"``:
categorical cross-entropy, locked decision) and ``kind="accuracy"`` returns
raw accuracy which is passed directly. Loss-based fitness can minimise other
losses (e.g. squared ``"hinge"``) via ``loss``. Every call evaluates the whole
population on the same batch of the fit split.

Batch schedule: ``"random"`` (default) draws a batch without replacement per
call; ``"cycle"`` walks the fit split as a fixed sequence of contiguous
batches of ``batch_size`` (last one partial) and wraps around, i.e. the
mini-batch tournament protocol of the mgab microbial-GA reference.
"""

import numpy as np
import tensorflow as tf

from bes import mlp


class BNNFitness:
    """(P, D) bool + (P, D_b) float callable -> (P,) float32 fitness."""

    def __init__(self, x, y, kind="loss", loss="ce", batch_size=4096, seed=None, shapes=None,
                 chunk_size=8192, schedule="random", scale="fixed", activation="sign",
                 pop_chunk=None):
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if pop_chunk is not None and int(pop_chunk) <= 0:
            raise ValueError(f"pop_chunk must be positive, got {pop_chunk}")
        if schedule not in ("random", "cycle"):
            raise ValueError(f"unknown batch schedule: {schedule}")
        self.x = x
        self.y = y
        self.kind = kind
        self.loss = loss
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.shapes = shapes if shapes is not None else mlp.weight_shapes()
        self.chunk_size = chunk_size
        self.schedule = schedule
        self.scale = scale
        self.activation = activation
        self.pop_chunk = pop_chunk
        self._cycle = 0
        if batch_size >= x.shape[0]:
            self.batch_size = x.shape[0]

    def _select_indices(self):
        n = int(self.x.shape[0])
        if self.schedule == "cycle":
            num_batches = -(-n // self.batch_size)
            start = (self._cycle * self.batch_size) % n
            self._cycle = (self._cycle + 1) % num_batches
            return tf.constant(np.arange(start, min(start + self.batch_size, n)), dtype=tf.int32)
        if self.schedule == "random":
            return tf.constant(
                self.rng.choice(n, size=self.batch_size, replace=False), dtype=tf.int32
            )
        raise ValueError(f"unknown batch schedule: {self.schedule}")

    def __call__(self, bits, bias):
        idx = self._select_indices()
        loss, acc = mlp.population_metrics(
            bits, tf.gather(self.x, idx, axis=0), tf.gather(self.y, idx, axis=0),
            chunk_size=self.chunk_size, shapes=self.shapes,
            loss=self.loss, bias=bias, scale=self.scale, activation=self.activation,
            pop_chunk=self.pop_chunk,
        )
        if self.kind == "accuracy":
            return acc
        if self.kind == "loss":
            return -loss
        raise ValueError(f"unknown fitness kind: {self.kind}")
