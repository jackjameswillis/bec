"""Binary MLP architecture and batched population evaluation."""

import math

import tensorflow as tf

INPUT_DIM = 784
HIDDEN_UNITS = 64
NUM_CLASSES = 10


def weight_shapes(input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES):
    hidden = (hidden_units,) if isinstance(hidden_units, int) else tuple(hidden_units)
    shapes = []
    prev = input_dim
    for units in hidden:
        shapes.append((prev, units))
        prev = units
    if not shapes:
        shapes = [(input_dim, num_classes)]
    else:
        shapes.append((hidden[-1], num_classes))
    return tuple(shapes)


def num_bits(input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES):
    return sum(in_dim * out_dim for in_dim, out_dim in weight_shapes(input_dim, hidden_units, num_classes))


def num_bias(input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES):
    """Total number of continuous bias parameters across all layers."""
    return sum(int(out_dim) for _, out_dim in weight_shapes(input_dim, hidden_units, num_classes))


def split_weights(bits, shapes=None):
    if shapes is None:
        shapes = weight_shapes()
    z = 2.0 * tf.cast(bits, tf.float32) - 1.0
    weights = []
    offset = 0
    for in_dim, out_dim in shapes:
        size = in_dim * out_dim
        weights.append(tf.reshape(z[:, offset:offset + size], (-1, in_dim, out_dim)))
        offset += size
    return z, weights


def split_biases(bias, shapes=None):
    """Split a flat bias vector into per-layer bias tensors.

    ``bias`` is either a flat ``(D_b,)`` vector for a single net or a rank-2
    ``(P, D_b)`` tensor for ``P`` individuals; the result is a list with one
    bias tensor per weight layer, shaped ``(P, 1, out_dim)`` (or ``(out_dim,)``
    for a single net) so it broadcasts over the population and batch axes.
    """
    if shapes is None:
        shapes = weight_shapes()
    bias = tf.cast(bias, tf.float32)
    parts = []
    offset = 0
    for _, out_dim in shapes:
        size = int(out_dim)
        part = bias[..., offset:offset + size]
        parts.append(tf.reshape(part, (-1, 1, size)) if int(bias.shape.rank) == 2 else part)
        offset += size
    return parts


def quantise_input(x):
    return 1.0 - 2.0 * tf.cast(x < 0.0, tf.float32)


def _scale(in_dim: int, mode: str) -> float:
    """Per-layer pre-activation scale: ``"fixed"`` = 1/sqrt(in_dim) (sign units),
    ``"he"`` = sqrt(2/in_dim) (He-style, variance-stable for binary weights with
    relu hidden units), ``"none"`` = 1."""
    if mode == "fixed":
        return 1.0 / math.sqrt(in_dim)
    if mode == "he":
        return math.sqrt(2.0 / in_dim)
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown scale mode: {mode}")


def _hidden_activation(s, kind: str):
    if kind == "sign":
        return tf.math.sign(s)
    if kind == "relu":
        return tf.nn.relu(s)
    raise ValueError(f"unknown hidden activation: {kind}")


def _forward_chunk(weights, biases, x, scale="fixed", activation="sign"):
    if biases is None:
        biases = [tf.zeros((int(w.shape[2]),)) for w in weights]

    if len(weights) == 1:
        return tf.einsum("bi,pih->pbh", x, weights[0]) * _scale(int(x.shape[-1]), scale) + biases[0]

    activation_out = _hidden_activation(
        tf.einsum("bi,pih->pbh", x, weights[0]) * _scale(int(x.shape[-1]), scale) + biases[0],
        activation,
    )
    for weight, bias in zip(weights[1:-1], biases[1:-1]):
        s = tf.einsum("pbh,phk->pbk", activation_out, weight)
        activation_out = _hidden_activation(
            s * _scale(int(tf.shape(activation_out)[-1]), scale) + bias, activation
        )

    final = tf.einsum("pbh,phk->pbk", activation_out, weights[-1])
    return final * _scale(int(tf.shape(activation_out)[-1]), scale) + biases[-1]


def categorical_cross_entropy(s, y):
    labels = tf.argmax(y, axis=-1, output_type=tf.int32)
    labels = tf.broadcast_to(labels[tf.newaxis], tf.shape(s)[:2])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=s)
    return tf.reduce_mean(ce, axis=1)


def squared_hinge(s, y):
    score = tf.reduce_sum(tf.cast(y, s.dtype) * s, axis=-1)
    margin = tf.maximum(tf.zeros_like(score), 1.0 - score)
    return tf.reduce_mean(tf.square(margin), axis=1)


def bnn_loss(s, y, kind: str):
    if kind == "ce":
        return categorical_cross_entropy(s, y)
    if kind == "hinge":
        return squared_hinge(s, y)
    raise ValueError(f"unknown loss kind: {kind}")


def _population_metrics_full(bits, x, y, chunk_size, shapes, loss, bias,
                             scale, activation):
    _, weights = split_weights(bits, shapes)
    biases = split_biases(bias, shapes) if bias is not None else None
    n = int(x.shape[0])
    loss_sum = tf.zeros([bits.shape[0]], tf.float32)
    correct_sum = tf.zeros_like(loss_sum)
    for start in range(0, n, chunk_size):
        xb, yb = x[start : start + chunk_size], y[start : start + chunk_size]
        s = _forward_chunk(weights, biases, xb, scale, activation)
        loss_sum += bnn_loss(s, yb, loss) * tf.cast(tf.shape(xb)[0], tf.float32)
        pred = tf.argmax(s, axis=-1)
        correct_sum += tf.reduce_sum(
            tf.cast(tf.equal(pred, tf.argmax(yb, axis=1)), tf.float32), axis=1
        )
    return loss_sum / tf.cast(n, tf.float32), correct_sum / tf.cast(n, tf.float32)


def population_metrics(bits, x, y, chunk_size=8192, shapes=None, loss="ce", bias=None,
                       scale="fixed", activation="sign", pop_chunk=None):
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if pop_chunk is not None and int(pop_chunk) <= 0:
        raise ValueError(f"pop_chunk must be positive, got {pop_chunk}")
    if shapes is None:
        shapes = weight_shapes()
    p = int(bits.shape[0])
    if pop_chunk is None or pop_chunk >= p:
        return _population_metrics_full(bits, x, y, chunk_size, shapes, loss,
                                        bias, scale, activation)
    bias_rows = bias if bias is not None else None
    parts = [
        _population_metrics_full(bits[i : i + pop_chunk], x, y, chunk_size,
                                 shapes, loss,
                                 None if bias_rows is None else bias_rows[i : i + pop_chunk],
                                 scale, activation)
        for i in range(0, p, pop_chunk)
    ]
    loss_all, acc_all = zip(*parts)
    return tf.concat(loss_all, axis=0), tf.concat(acc_all, axis=0)


def net_probs(bits, x, shapes=None, bias=None, scale="fixed", activation="sign"):
    """Softmax probabilities of one net."""
    _, weights = split_weights(tf.expand_dims(bits, 0), shapes)
    biases = split_biases(bias, shapes) if bias is not None else None
    return tf.nn.softmax(_forward_chunk(weights, biases, x, scale, activation), axis=-1)


net_logits = net_probs
