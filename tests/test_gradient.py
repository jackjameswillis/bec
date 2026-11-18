"""Exact-gradient test for the soft Bernoulli ES on a tiny binary MLP.

Enumerates every bit pattern of a 2->2->1 binary net, forms the exact
expected fitness under the Bernoulli(theta) distribution, and checks:

1. the closed-form per-bit score ``(x - sigma(theta))`` against a numerical
   derivative of the exact expected fitness (population-summed analytic
   gradient == P * grad E[f]);
2. ``SoftES.update`` on a constructed finite population matches the
   population-summed estimator including the mean-fitness baseline and lr.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes import mlp
from bes.algs.es import SoftES

SHAPES = ((2, 2), (2, 2))


def all_pattern_losses(bits_mat):
    _, weights = mlp.split_weights(bits_mat, shapes=SHAPES)
    biases = mlp.split_biases(
        tf.zeros((tf.shape(bits_mat)[0], mlp.num_bias(2, 2, 2)), tf.float32), SHAPES
    )
    s = mlp._forward_chunk(weights, biases, mlp.quantise_input(x_batch))
    return mlp.bnn_loss(s, y_batch, "ce").numpy()


def main():
    global x_batch, y_batch
    rng = np.random.default_rng(3)
    x_batch = rng.normal(size=(8, 2)).astype("float32")
    y_batch = tf.cast(tf.one_hot((x_batch[:, 0] > 0).astype("int64"), 2), tf.float32)

    num_bits = mlp.num_bits(2, 2, 2)
    all_bits = np.array(list(itertools.product([0, 1], repeat=num_bits)), dtype=bool)
    bits_tensor = tf.convert_to_tensor(all_bits)

    theta0 = np.array([0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.05, -0.15])
    s = 1.0 / (1.0 + np.exp(-theta0))
    losses = all_pattern_losses(bits_tensor).astype(np.float64)
    fit = -losses

    probs = np.array(
        [np.prod(s ** b * (1 - s) ** (1 - b)) for b in all_bits.astype(np.float64)]
    )
    probs /= probs.sum()

    def expected_fitness(theta):
        theta = np.asarray(theta, dtype=np.float64)
        sv = 1.0 / (1.0 + np.exp(-theta))
        ls = all_pattern_losses(bits_tensor).astype(np.float64)
        pr = np.array([np.prod(sv ** b * (1 - sv) ** (1 - b)) for b in all_bits])
        pr /= pr.sum()
        return float(-(pr * ls).sum())

    eps = 1e-4
    numeric = np.array(
        [
            (expected_fitness(theta0 + eps * e_j) - expected_fitness(theta0 - eps * e_j))
            / (2 * eps)
            for e_j in np.eye(num_bits)
        ]
    )

    b_exact = float((probs * fit).sum())
    analytic_exact = (probs[:, None] * ((fit - b_exact)[:, None] *
                      (all_bits.astype(np.float64) - s))).sum(axis=0)
    np.testing.assert_allclose(analytic_exact, numeric, rtol=1e-4, atol=1e-5)

    es = SoftES(num_bits, pop_size := 5, p_init=0.5, seed=0)
    xs = all_bits[:pop_size].astype(np.float64)
    fs = fit[:pop_size]
    es.population = tf.convert_to_tensor(all_bits[:pop_size], dtype=tf.bool)
    es.fitness = tf.convert_to_tensor(fs.astype(np.float32))
    es.theta.assign(theta0.astype(np.float32))

    before = es.theta.numpy().copy()
    lr = 0.01
    es.update(lr)
    delta_tf = es.theta.numpy() - before
    delta_ref = lr * ((fs - fs.mean())[:, None] * (xs - s)).sum(axis=0)
    np.testing.assert_allclose(delta_tf, delta_ref, rtol=1e-5, atol=1e-7)

    # 2b. continuous bias: Gaussian location-family REINFORCE step
    es_b = SoftES(num_bits, pop_size, p_init=0.5, seed=0, num_bias=mlp.num_bias(2, 2, 2), sigma=0.25)
    es_b.population = tf.convert_to_tensor(all_bits[:pop_size], dtype=tf.bool)
    es_b.fitness = tf.convert_to_tensor(fs.astype("float32"))
    es_b.bias = tf.convert_to_tensor(np.random.default_rng(11).normal(size=(pop_size, mlp.num_bias(2, 2, 2))).astype("float32"), tf.float32)
    es_b.theta_bias.assign(np.zeros(mlp.num_bias(2, 2, 2), "float32"))
    before_b = es_b.theta_bias.numpy().copy()
    es_b.update(lr)
    delta_b = es_b.theta_bias.numpy() - before_b
    delta_b_ref = lr * ((fs - fs.mean())[:, None] * (es_b.bias.numpy() - np.zeros_like(es_b.bias.numpy())) / (0.25 ** 2)).sum(axis=0)
    np.testing.assert_allclose(delta_b, delta_b_ref, rtol=1e-5, atol=1e-7)

    print("exact REINFORCE gradient OK")


if __name__ == "__main__":
    main()
