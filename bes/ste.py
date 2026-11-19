"""STE baseline: fully-binary MLP on MNIST (larq), ported from
``/data/ste/train_mnist_bnn.py``.

Architecture follows the original code (not its stale docstring): 784 -> 64
hidden -> 10 with 1-bit weights by default (``hidden_units``/``hidden_layers``
are configurable). ``QuantDense`` layers use ``ste_sign``
kernel quantizers and the ``weight_clip`` constraint; by default the first
layer's ``ste_sign`` input quantizer sign-binarises the (normalised,
full-precision float) pixels exactly as in the original script — pass
``input_quant=False`` (``run_ste.py --no-input-quant``) to feed the first
layer the raw normalised pixels instead. Training: plain SGD without
momentum, categorical cross-entropy, no label smoothing.

Locked design decision "no biases for now" is applied as the default
(``use_bias=False``) so STE and BES optimise the same hypothesis
space; pass ``use_bias=True`` to add biases after each per-layer
pre-activation scale, matching the ES forward (with ``scale="none"`` this is
the original script's bias placement).
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy

from larq.layers import QuantDense

from bes import data, mlp

HIDDEN_UNITS = mlp.HIDDEN_UNITS
NUM_CLASSES = data.NUM_CLASSES
INPUT_DIM = data.INPUT_DIM
INPUT_QUANTIZER = "ste_sign"
KERNEL_QUANTIZER = "ste_sign"


class Bias(tf.keras.layers.Layer):
    """Trainable per-output bias added after the pre-activation scale."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        super().build(input_shape)
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.bias


def build_model(learning_rate=1e-3, use_bias=False, hidden_units=HIDDEN_UNITS,
                hidden_layers=1, scale="none", optimizer="sgd", input_quant=True):
    c_in = mlp._scale(INPUT_DIM, scale)
    c_h = mlp._scale(hidden_units, scale)
    layers = [Input(shape=(INPUT_DIM,))]
    for i in range(hidden_layers):
        layers.append(
            QuantDense(
                hidden_units,
                # sign-binarise the (normalised) input pixels exactly once,
                # as in the original script; deeper QuantDense layers take
                # the full-precision (scaled) relu activations
                input_quantizer=INPUT_QUANTIZER if i == 0 and input_quant else None,
                kernel_quantizer=KERNEL_QUANTIZER,
                kernel_constraint="weight_clip",
                activation=None if use_bias else "relu",
                use_bias=False,
            )
        )
        if scale != "none":
            # relu-positive-homogeneity makes post-relu scaling equivalent to
            # the ES forward's per-layer pre-activation scale
            layers.append(Lambda(lambda t, c=c_in if i == 0 else c_h: t * c))
        if use_bias:
            layers.append(Bias(hidden_units))
            layers.append(tf.keras.layers.Activation("relu"))
    layers.append(
        QuantDense(
            NUM_CLASSES,
            kernel_quantizer=KERNEL_QUANTIZER,
            kernel_constraint="weight_clip",
            use_bias=False,
        )
    )
    if scale != "none":
        layers.append(Lambda(lambda t, c=c_h: t * c))
    if use_bias:
        layers.append(Bias(NUM_CLASSES))
    layers.append(tf.keras.layers.Activation("softmax"))
    model = Sequential(layers)
    if optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate, momentum=0.0)
    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    model.compile(
        optimizer=opt,
        loss=categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def train(model, x_fit, y_fit, x_val, y_val, batch_size=512, epochs=25,
          callbacks=None):
    return model.fit(
        x_fit,
        y_fit,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
