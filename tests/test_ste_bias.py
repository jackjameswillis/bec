"""STE bias-order parity against the ES forward.

With ``use_bias=True`` the STE must add each layer's bias after the
pre-activation scale, exactly like ``bes/mlp._forward_chunk``. This test fixes
the same binary weights and biases in both models and compares the softmax
outputs for every scale mode and for one and two hidden layers.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from bes import mlp, ste


def set_weights(model, bits, bias, shapes):
    kernels = [layer for layer in model.layers
               if isinstance(layer, ste.QuantDense)]
    biases = [layer for layer in model.layers if isinstance(layer, ste.Bias)]
    _, weights = mlp.split_weights(bits[None, :], shapes)
    for layer, weight in zip(kernels, weights):
        layer.set_weights([weight[0].numpy()])
    offset = 0
    for layer, (_, out_dim) in zip(biases, shapes):
        layer.set_weights([bias[offset:offset + out_dim]])
        offset += out_dim
    assert offset == len(bias)


def check(scale, layers, rng):
    units = 4
    shapes = mlp.weight_shapes(hidden_units=(units,) * layers)
    num_bits = mlp.num_bits(hidden_units=(units,) * layers)
    num_bias = mlp.num_bias(hidden_units=(units,) * layers)
    bits = rng.integers(0, 2, size=num_bits).astype(np.float32)
    bias = rng.normal(size=num_bias).astype(np.float32)
    x = rng.normal(size=(3, mlp.INPUT_DIM)).astype(np.float32)

    model = ste.build_model(use_bias=True, hidden_units=units,
                            hidden_layers=layers, scale=scale,
                            input_quant=False)
    model(tf.zeros((1, mlp.INPUT_DIM), dtype=tf.float32))
    set_weights(model, bits, bias, shapes)
    got = model(tf.constant(x), training=False).numpy()

    _, weights = mlp.split_weights(bits[None, :], shapes)
    ref = mlp.net_logits(bits, tf.constant(x), shapes=shapes,
                         bias=tf.constant(bias), scale=scale,
                         activation="relu").numpy()[0]
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    print(f"STE bias parity OK: scale={scale} layers={layers}")


def main():
    rng = np.random.default_rng(11)
    for scale in ("none", "fixed", "he"):
        for layers in (1, 2):
            check(scale, layers, rng)
    print("STE bias ordering matches the ES forward")


if __name__ == "__main__":
    main()
