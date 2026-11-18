"""Parity test: generic BNN forward/loss vs hand-rolled binary reference.

Builds a tiny binary MLP by hand (sign hidden activations, selectable
pre-activation scale: fixed/he/none, softmax final output) and compares
against ``bes.mlp`` running the same individual through the population
pipeline. Also checks the loss kinds (cross-entropy and squared hinge)
against hand-computed values.
"""

import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes import mlp


def ref_scale(mode, d):
    """Reference per-layer pre-activation scale (mirrors bes.mlp._scale)."""
    if mode == "fixed":
        return 1.0 / math.sqrt(int(d))
    if mode == "he":
        return math.sqrt(2.0 / int(d))
    return 1.0


def build_reference(w1, w2, b1, b2, scale="fixed", relu=False):
    def forward(x):
        k = lambda d: ref_scale(scale, d)
        a = tf.nn.relu if relu else tf.math.sign
        h = a(tf.einsum("bi,pih->pbh", x, w1) * k(int(x.shape[-1])) + b1)
        out = tf.einsum("pbh,phk->pbk", h, w2) * k(int(tf.shape(h)[-1])) + b2
        return tf.nn.softmax(out, axis=-1)

    return forward


def build_reference_multi(weights, biases, scale="fixed", relu=False):
    """Hand-rolled forward for arbitrary depth (multi hidden layers)."""
    def forward(x):
        k = lambda d: ref_scale(scale, d)
        a = tf.nn.relu if relu else tf.math.sign
        h = a(tf.einsum("bi,pih->pbh", x, weights[0]) * k(int(x.shape[-1])) + biases[0])
        for w, b in zip(weights[1:-1], biases[1:-1]):
            s = tf.einsum("pbh,phk->pbk", h, w) * k(int(tf.shape(h)[-1])) + b
            h = a(s)
        out = tf.einsum("pbh,phk->pbk", h, weights[-1]) * k(int(tf.shape(h)[-1])) + biases[-1]
        return tf.nn.softmax(out, axis=-1)

    return forward


def main():
    rng = np.random.default_rng(7)
    input_dim, hidden, classes = 7, 5, 3
    shapes = ((input_dim, hidden), (hidden, classes))
    bits = rng.random(mlp.num_bits(input_dim, hidden, classes)) > 0.5

    bits_tensor = tf.convert_to_tensor(bits)
    _, weights = mlp.split_weights(tf.expand_dims(bits_tensor, 0), shapes)
    w1, w2 = weights
    bias = tf.zeros((1, mlp.num_bias(input_dim, hidden, classes)), tf.float32)
    b1, b2 = mlp.split_biases(bias, shapes)

    x = rng.normal(size=(64, input_dim)).astype("float32")
    ref = build_reference(w1, w2, b1, b2)(tf.convert_to_tensor(x))
    ours = mlp.net_logits(bits, tf.convert_to_tensor(x), shapes=shapes, bias=bias)

    assert ref.shape == ours.shape, (ref.shape, ours.shape)
    np.testing.assert_allclose(ref.numpy(), ours.numpy(), rtol=1e-5, atol=1e-6)
    print("forward parity OK: max abs diff", float(np.abs(ref.numpy() - ours.numpy()).max()))

    ref_nos = build_reference(w1, w2, b1, b2, scale="none")(tf.convert_to_tensor(x))
    ours_nos = mlp.net_logits(bits, tf.convert_to_tensor(x), shapes=shapes, bias=bias,
                              scale="none")
    np.testing.assert_allclose(ref_nos.numpy(), ours_nos.numpy(), rtol=1e-5, atol=1e-6)
    print("no-scale forward parity OK: max abs diff",
          float(np.abs(ref_nos.numpy() - ours_nos.numpy()).max()))

    ref_he = build_reference(w1, w2, b1, b2, scale="he", relu=True)(
        tf.convert_to_tensor(x))
    ours_he = mlp.net_logits(bits, tf.convert_to_tensor(x), shapes=shapes, bias=bias,
                             scale="he", activation="relu")
    np.testing.assert_allclose(ref_he.numpy(), ours_he.numpy(), rtol=1e-5, atol=1e-6)
    print("relu he-scale forward parity OK: max abs diff",
          float(np.abs(ref_he.numpy() - ours_he.numpy()).max()))

    ref_relu = build_reference(w1, w2, b1, b2, scale="none", relu=True)(
        tf.convert_to_tensor(x))
    ours_relu = mlp.net_logits(bits, tf.convert_to_tensor(x), shapes=shapes, bias=bias,
                               scale="none", activation="relu")
    np.testing.assert_allclose(ref_relu.numpy(), ours_relu.numpy(), rtol=1e-5, atol=1e-6)
    print("relu no-scale forward parity OK: max abs diff",
          float(np.abs(ref_relu.numpy() - ours_relu.numpy()).max()))

    hidden_units = (4, 4, 3)
    shapes3 = mlp.weight_shapes(input_dim, hidden_units, classes)
    assert shapes3 == ((input_dim, 4), (4, 4), (4, 3), (3, classes)), shapes3
    bits3 = rng.random(mlp.num_bits(input_dim, hidden_units, classes)) > 0.5
    bits3_tensor = tf.convert_to_tensor(bits3)
    _, weights3 = mlp.split_weights(tf.expand_dims(bits3_tensor, 0), shapes3)
    bias3 = tf.cast(rng.normal(size=mlp.num_bias(input_dim, hidden_units, classes)) * 0.1,
                    tf.float32)
    biases3 = mlp.split_biases(bias3, shapes3)
    ref3 = build_reference_multi(weights3, biases3)(tf.convert_to_tensor(x))
    ours3 = mlp.net_logits(bits3, tf.convert_to_tensor(x), shapes=shapes3, bias=bias3)
    np.testing.assert_allclose(ref3.numpy(), ours3.numpy(), rtol=1e-5, atol=1e-6)
    print("multi-hidden-layer (3 layers) forward parity OK: max abs diff",
          float(np.abs(ref3.numpy() - ours3.numpy()).max()))

    ref3_nos = build_reference_multi(weights3, biases3, scale="none", relu=True)(
        tf.convert_to_tensor(x))
    ours3_nos = mlp.net_logits(bits3, tf.convert_to_tensor(x), shapes=shapes3, bias=bias3,
                               scale="none", activation="relu")
    np.testing.assert_allclose(ref3_nos.numpy(), ours3_nos.numpy(), rtol=1e-5, atol=1e-6)
    print("multi-hidden-layer relu no-scale forward parity OK: max abs diff",
          float(np.abs(ref3_nos.numpy() - ours3_nos.numpy()).max()))

    ref3_he = build_reference_multi(weights3, biases3, scale="he", relu=True)(
        tf.convert_to_tensor(x))
    ours3_he = mlp.net_logits(bits3, tf.convert_to_tensor(x), shapes=shapes3, bias=bias3,
                              scale="he", activation="relu")
    np.testing.assert_allclose(ref3_he.numpy(), ours3_he.numpy(), rtol=1e-5, atol=1e-6)
    print("multi-hidden-layer relu he-scale forward parity OK: max abs diff",
          float(np.abs(ref3_he.numpy() - ours3_he.numpy()).max()))

    try:
        mlp._scale(4, "bogus")
        raise AssertionError("should have raised")
    except ValueError as e:
        print("unknown scale mode raises OK:", e)

    s = tf.constant(rng.normal(size=(2, 4, classes)).astype("float32"))
    y = tf.cast(tf.one_hot(np.array([0, 2, 1, 2]), classes), tf.float32)

    ce = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.broadcast_to(y[tf.newaxis], tf.shape(s)), logits=s
    )
    ce = tf.reduce_mean(ce, axis=1)
    np.testing.assert_allclose(mlp.bnn_loss(s, y, "ce").numpy(), ce.numpy(), rtol=1e-6)

    score = tf.reduce_sum(tf.cast(y, s.dtype) * s, axis=-1)
    margin = tf.maximum(tf.zeros_like(score), 1.0 - score)
    hinge = tf.reduce_mean(tf.square(margin), axis=1)
    np.testing.assert_allclose(mlp.bnn_loss(s, y, "hinge").numpy(), hinge.numpy(), rtol=1e-6)
    print("loss kinds OK: ce and squared hinge")

    try:
        mlp.bnn_loss(s, y, "bogus")
        raise AssertionError("should have raised")
    except ValueError as e:
        print("unknown loss kind raises OK:", e)


if __name__ == "__main__":
    main()
