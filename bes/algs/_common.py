"""Shared machinery for the population-based search algorithms."""

import tensorflow as tf


def row_chunk_size(num_bits, budget=1_000_000_000, max_rows=128):
    """Rows per noise chunk keeping ``(rows, D)`` float temporaries ~``budget``."""
    return min(int(max_rows), max(1, int(budget) // (12 * int(num_bits))))


def fitness_utility(f, shaping):
    """Per-individual utility from a population fitness vector.

    ``"none"``: ``f - mean(f)``. ``"standardize"``: population z-score.
    ``"rank"``: centered ordinal ranks in ``[-0.5, 0.5]`` — ``r / (P - 1) - 0.5``
    with ``r`` the 0-based sort position (OpenAI-ES convention; a stable sort
    breaks ties by population index). ``P == 1`` gives zero utility.
    """
    if shaping == "none":
        return f - tf.reduce_mean(f)
    if shaping == "standardize":
        return (f - tf.reduce_mean(f)) / (tf.math.reduce_std(f) + 1e-12)
    if shaping == "rank":
        n = tf.shape(f)[0]
        p = tf.cast(n, tf.float32)
        if float(p) <= 1.0:
            return tf.zeros_like(f)
        order = tf.argsort(f, stable=True)
        ranks = tf.math.invert_permutation(order)
        return tf.cast(ranks, tf.float32) / (p - 1.0) - 0.5
    raise ValueError(f"unknown fitness shaping: {shaping}")


def fitness_stats(f):
    """Scalar stats + argmax/argmin of the current population fitness."""
    if f is None:
        raise RuntimeError("fitness_stats() called before evaluate()")
    return dict(
        f_mean=float(tf.reduce_mean(f)),
        f_std=float(tf.math.reduce_std(f)),
        f_min=float(tf.reduce_min(f)),
        f_max=float(tf.reduce_max(f)),
        f_argmax=int(tf.argmax(f)),
        f_argmin=int(tf.argmin(f)),
    )


def entropy_and_commit(probs):
    """Mean per-bit binary entropy (bits) and committed fraction of ``probs``."""
    s = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)
    h = -(s * tf.math.log(s)
          + (1.0 - s) * tf.math.log(1.0 - s)) / tf.math.log(2.0)
    committed = tf.cast(
        tf.logical_or(s < 0.01, s > 1.0 - 0.01), tf.float32)
    return h, committed


def diagnostics_dict(probs, theta, extra=None, layer_sizes=None):
    """Scalar convergence stats; ``extra`` adds an algorithm-specific fraction."""
    h, committed = entropy_and_commit(probs)
    out = dict(
        entropy_bits=float(tf.reduce_mean(h)),
        committed_frac=float(tf.reduce_mean(committed)),
        theta_std=float(tf.math.reduce_std(theta)),
    )
    if extra is not None:
        out["flip_frac"] = float(tf.reduce_mean(extra))
    if layer_sizes:
        out["entropy_bits_layer"] = [
            float(tf.reduce_mean(part)) for part in tf.split(h, layer_sizes, axis=0)]
        out["committed_frac_layer"] = [
            float(tf.reduce_mean(part))
            for part in tf.split(committed, layer_sizes, axis=0)]
        if extra is not None:
            out["flip_frac_layer"] = [
                float(tf.reduce_mean(part))
                for part in tf.split(extra, layer_sizes, axis=0)]
    return out


def packed_diagnostics(probs, theta, f, extra=None, layer_sizes=None):
    """All per-iteration diagnostics as one float32 tensor (single device sync).

    Layout: ``[entropy_bits, committed_frac, (flip_frac), theta_std, f_mean,
    f_std, f_min, f_max, f_argmax, f_argmin]`` followed by per-layer
    ``[entropy_bits_layer_i, committed_frac_layer_i, (flip_frac_layer_i)]``
    groups when ``layer_sizes`` is given.
    """
    if f is None:
        raise RuntimeError("diagnostics_packed() called before evaluate()")
    h, committed = entropy_and_commit(probs)
    parts = [tf.reduce_mean(h), tf.reduce_mean(committed)]
    if extra is not None:
        parts.append(tf.reduce_mean(extra))
    parts.extend([
        tf.math.reduce_std(theta),
        tf.reduce_mean(f),
        tf.math.reduce_std(f),
        tf.reduce_min(f),
        tf.reduce_max(f),
        tf.cast(tf.argmax(f), tf.float32),
        tf.cast(tf.argmin(f), tf.float32),
    ])
    if layer_sizes:
        splits = [tf.split(h, layer_sizes, axis=0),
                  tf.split(committed, layer_sizes, axis=0)]
        if extra is not None:
            splits.append(tf.split(extra, layer_sizes, axis=0))
        for group in zip(*splits):
            parts.extend(tf.reduce_mean(part) for part in group)
    return tf.stack(parts)


def adam_step(param, grad, m, v, t, beta1, beta2, eps, lr):
    """One bias-corrected Adam step on a population-summed gradient."""
    m.assign(beta1 * m + (1.0 - beta1) * grad)
    v.assign(beta2 * v + (1.0 - beta2) * grad ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    param.assign_add(tf.cast(lr, tf.float32) * m_hat / (tf.sqrt(v_hat) + eps))


class ESCommon:
    """Mixin for the shared fitness / Adam / diagnostics bookkeeping of the ESs."""

    def _utility(self):
        return fitness_utility(self.fitness, self.shaping)

    def _adam_step(self, param, grad, m, v, lr):
        adam_step(param, grad, m, v, self._adam_t, self.beta1, self.beta2,
                  self.eps, lr)

    def fitness_mean(self):
        return tf.reduce_mean(self.fitness)

    def fitness_best(self):
        return tf.reduce_max(self.fitness)

    def fitness_stats(self):
        return fitness_stats(self.fitness)

    def step(self, lr, fitness_fn):
        self.sample()
        self.evaluate(fitness_fn)
        self.update(lr)
