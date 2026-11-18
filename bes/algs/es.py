"""Soft evolutionary strategy for binary individuals, logit parameterisation.

Per bit: ``p(x=1 | theta) = sigmoid(theta)``. Individuals are sign vectors
``z = 2x - 1`` with expectation ``E[z] = 2 sigmoid(theta) - 1`` (see
``/data/reinforce_maths.tex``). The Bernoulli log-gradient has the closed form

    grad_theta log p(x | theta) = x - sigmoid(theta),

equivalently ``(z - (2 sigmoid(theta) - 1)) / 2`` under the sign mapping; the
denominator ``theta (1 - theta)`` of the comb-exp probability parameterisation
cancels exactly under this sigmoid reparameterisation.

REINFORCE update with mean-fitness baseline (locked decision), in the
population-summed estimator form of comb-exp ``algs/es.py``:

    theta <- theta + lr * sum_i (f_i - mean(f)) * (x_i - sigmoid(theta))

where the population sum scales the effective learning rate by P. Fitness is
maximised, matching comb-exp convention: pass loss fitness negated.

Perturbations come from a pre-generated table instead of random numbers
drawn during optimisation. The constructor fills ``theta``-independent
uniforms ``u in [0, 1)^(T, D)`` in one call; every generation gathers the
next circular block of rows and binarises them against the current
``sigmoid(theta)``, so the table is reusable at every theta and the
generation samples stay exactly ``Bernoulli(sigma(theta))`` (no RNG in the
loop, rows wrap around after T rows). ``table_rows=None`` keeps the legacy
streaming sampler, which draws fresh uniform row-chunks each step. Sampling
under a fixed seed is deterministic; a table run is not bit-identical to a
streaming run (philox block counts differ between one large and many small
uniform calls), only statistically equivalent.

With ``antithetic=True`` (default off), each generation consumes
``ceil(P / 2)`` rows and pairs each row ``u`` with its mirror ``1 - u``:
the second half of the population is binarised from the mirrored noise,
still exactly ``Bernoulli(sigma(theta))``. At ``p_init = 0.5`` (Bernoulli
rate 0.5) pair mates are exact complement bits; the pair-projected score
``[(f(u) - f(1-u)) (x - s)]`` has reduced variance (cf. antithetic pairs in
OpenAI-ES). An odd population appends one unpaired row.

Continuous layer biases (hidden + output) ride along in the same population:
weight bits keep the Bernoulli logit distribution while biases are sampled
from a Gaussian with mean ``theta_bias`` and fixed perturbation scale
``sigma``. Both coordinates are updated by the same population-summed
REINFORCE step — Bernoulli score ``(z - s) / 2`` for weights, Gaussian score
``(b - theta_bias) / sigma^2`` for biases. The bias coordinate may take its
own learning rate (``bias_lr``); ``None`` (default) falls back to the shared
``lr`` — the score is untouched, only the step size differs.

Fitness shaping (``shaping``) replaces the raw advantage ``f_i - mean(f)``
with a bounded per-individual utility shared by both coordinate gradients
(Salimans et al. 1703.03864): ``"none"`` = raw advantage (unbiased REINFORCE),
``"standardize"`` = population z-score, ``"rank"`` = centered average ranks in
``[-0.5, 0.5]`` (tied fitnesses share average ranks; all-equal fitness gives
zeros). Shaped utilities depend on the whole population's fitness, so unlike
``"none"`` they are not unbiased REINFORCE baselines — they bound the step
size and decouple it from the fitness scale (e.g. exploding CE), at the cost
of estimator bias.

With ``adam=True`` the population-summed gradients are preprocessed by a
per-coordinate Adam rule (shared step counter, bias-corrected moments,
OpenAI-ES 1703.03864 style) instead of being added raw: each coordinate's
step becomes ``lr * m_hat / (sqrt(v_hat) + eps)``, bounded by ~``lr`` per
generation. This equalizes per-coordinate step sizes and — via the momentum
term — accumulates coherent signal across generations while the zero-mean
per-generation noise averages down. The raw path (``adam=False``, default)
is unchanged.

``update(lr, grad_chunk=k)`` splits the population reduction into ``k``-row
chunks so the ``(P, D)`` float temporaries stay bounded for large
populations on GPU (the weight-bit cast is also chunked). Default ``None``
reduces the whole population in one shot. Chunking only changes the
floating-point summation order, not the algorithm.

With ``natural=True`` the weight-bit gradient is divided by the diagonal
Bernoulli Fisher ``s(1 - s)`` (eps-damped) — the natural-gradient update of
Amari 1998 / NES (Wierstra et al. 2014) for the product-Bernoulli search
distribution. At ``p = 0.5`` this is a constant 4x rescale (lr-equivalent);
it diverges from the raw update as coordinates commit (``s -> 0`` or ``1``),
keeping frozen bits mobile. The logit parameterisation and the Gaussian
bias score ``(b - theta_bias) / sigma^2`` are already natural-gradient
equivalent by construction (the bias Fisher is ``1/sigma^2``).
"""

import tensorflow as tf

from bes.algs._common import (
    ESCommon, diagnostics_dict, packed_diagnostics, row_chunk_size,
)


class SoftES(ESCommon):
    """Bernoulli(logit) search distribution over binary individuals.

    Continuous bias coordinates (hidden + output layers) ride along in the
    same population: weights are sign bits with ``p(z=+1) = sigma(theta)`` while
    biases are Gaussian with mean ``theta_bias`` and fixed perturbation scale
    ``sigma``. Both are updated by the same population-summed REINFORCE step.
    """

    def __init__(self, num_bits, population_size, p_init=0.5, table_rows=1024,
                 antithetic=False, seed=None, num_bias=0, sigma=0.1,
                 shaping="none", bias_lr=None, adam=False,
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 natural=False, natural_eps=1e-2):
        p = float(p_init)
        if not 0.0 < p < 1.0:
            raise ValueError(f"p_init must be in (0, 1), got {p_init}")
        if float(sigma) < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if int(num_bits) <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")
        if int(population_size) <= 0:
            raise ValueError(
                f"population_size must be positive, got {population_size}")
        if int(num_bias) < 0:
            raise ValueError(f"num_bias must be non-negative, got {num_bias}")
        if shaping not in ("none", "standardize", "rank"):
            raise ValueError(f"unknown fitness shaping: {shaping}")
        if table_rows is not None and int(table_rows) < 0:
            raise ValueError(f"table_rows must be non-negative, got {table_rows}")
        logit = tf.constant(tf.math.log(p / (1.0 - p)), tf.float32)
        self.num_bits = int(num_bits)
        self.num_bias = int(num_bias)
        self.population_size = int(population_size)
        self.antithetic = bool(antithetic)
        self.sigma = float(sigma)
        self.shaping = str(shaping)
        self.bias_lr = None if bias_lr is None else float(bias_lr)
        self.adam = bool(adam)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._adam_t = 0
        self.natural = bool(natural)
        self.natural_eps = float(natural_eps)
        self.table_rows = None if not table_rows else int(table_rows)
        self.theta = tf.Variable(tf.fill([self.num_bits], logit), trainable=False)
        self.theta_bias = tf.Variable(tf.zeros([self.num_bias], tf.float32), trainable=False)
        self._m = tf.Variable(tf.zeros([self.num_bits], tf.float32), trainable=False)
        self._v = tf.Variable(tf.zeros([self.num_bits], tf.float32), trainable=False)
        self._m_bias = tf.Variable(tf.zeros([self.num_bias], tf.float32), trainable=False)
        self._v_bias = tf.Variable(tf.zeros([self.num_bias], tf.float32), trainable=False)
        self._gen = tf.random.Generator.from_seed(
            0 if seed is None else seed
        )
        self._table = (
            None if self.table_rows is None
            else self._gen.uniform(
                [self.table_rows, self.num_bits], minval=0.0, maxval=1.0
            )
        )
        self._rows_used = 0
        # adaptive noise row-chunk: bounds the (rows, D) float temporaries to
        # ~1 GB for wide nets (D = 2.9M -> 28 rows)
        self._sample_chunk = row_chunk_size(self.num_bits)
        self.population = None
        self.bias = tf.zeros([self.population_size, self.num_bias], tf.float32)
        self.fitness = None

    def probs(self):
        return tf.sigmoid(self.theta)

    def rows_per_generation(self):
        """Noise rows consumed per generation (antithetic pairs halve it)."""
        if self.antithetic:
            return self.population_size - self.population_size // 2
        return self.population_size

    def _noise_chunks(self, rows):
        """Yield (chunk_rows, D) theta-independent uniforms for one generation."""
        if self._table is not None:
            start = self._rows_used % self.table_rows
            idx = tf.range(start, start + rows) % self.table_rows
            self._rows_used += rows
            for i in range(0, rows, self._sample_chunk):
                yield tf.gather(self._table, idx[i:i + self._sample_chunk])
            return
        for i in range(0, rows, self._sample_chunk):
            k = min(self._sample_chunk, rows - i)
            yield self._gen.uniform([k, self.num_bits], minval=0.0, maxval=1.0)

    def sample(self):
        """Resample the population: (P, D) bool, p(1) = sigmoid(theta).

        Table mode consumes pre-generated noise rows but applies the current
        theta at binarisation; antithetic mode pairs each row ``u`` with
        ``1 - u`` for the second half of the population. Both table and
        streaming paths binarise row-chunks of at most 128, so the ``(rows, D)``
        float noise tensor never materializes whole (for populations larger
        than the chunk this changes the streaming RNG stream; statistically
        equivalent, not bit-identical to a single big draw).
        """
        s = self.probs()
        rows = self.rows_per_generation()
        if not self.antithetic:
            parts = [u < s for u in self._noise_chunks(rows)]
            self.population = (
                tf.concat(parts, axis=0) if len(parts) > 1 else parts[0])
            self.bias = self._sample_bias()
            return self.population

        k = self.population_size // 2
        paired_plus, mirrored, tail_plus = [], [], []
        pos = 0
        for u in self._noise_chunks(rows):
            m = int(u.shape[0])
            if pos < k:
                n = min(m, k - pos)
                paired_plus.append(u[:n] < s)
                mirrored.append((1.0 - u[:n]) < s)
            if pos + m > k:
                tail = u[max(0, k - pos):]
                if int(tail.shape[0]):
                    tail_plus.append(tail < s)
            pos += m
        self.population = tf.concat(paired_plus + mirrored + tail_plus, axis=0)
        self.bias = self._sample_bias()
        return self.population

    def _sample_bias(self):
        """(P, D_b) Gaussian bias values for the current generation.

        Antithetic pairs mirror the perturbation around the current mean
        ``theta_bias`` (``b <- 2 * theta_bias - b``), i.e. deviations sum to zero.
        ``sigma == 0`` disables bias learning: every individual carries the
        (frozen) mean ``theta_bias`` and ``update`` skips the bias gradient.
        """
        p, d = self.population_size, self.num_bias
        if self.sigma == 0.0:
            return tf.broadcast_to(self.theta_bias[None, :], [p, d])
        if self.antithetic:
            k = p // 2
            eps = self._gen.normal([k, d], stddev=self.sigma)
            bias = tf.concat([self.theta_bias + eps, self.theta_bias - eps], axis=0)
            if p % 2:
                tail = self._gen.normal([1, d], stddev=self.sigma)
                bias = tf.concat([bias, self.theta_bias + tail], axis=0)
            return bias[:p]
        eps = self._gen.normal([p, d], stddev=self.sigma)
        return self.theta_bias + eps

    def evaluate(self, fitness_fn):
        self.fitness = fitness_fn(self.population, self.bias)

    def update(self, lr, grad_chunk=None):
        s = self.probs()
        p = int(self.population.shape[0])
        u = self._utility()
        if grad_chunk is None or grad_chunk >= p:
            x = tf.cast(self.population, tf.float32)
            grad = tf.reduce_sum(u[:, None] * (x - s), axis=0)
        else:
            grad_chunk = int(grad_chunk)
            parts = []
            for i in range(0, p, grad_chunk):
                x = tf.cast(self.population[i:i + grad_chunk], tf.float32)
                parts.append(tf.reduce_sum(
                    u[i:i + grad_chunk, None] * (x - s), axis=0))
            grad = tf.add_n(parts)
        if self.natural:
            grad = grad / (s * (1.0 - s) + self.natural_eps)
        if self.num_bias > 0 and self.sigma > 0.0:
            grad_bias = tf.reduce_sum(
                u[:, None] * (self.bias - self.theta_bias[None, :]) / (self.sigma ** 2),
                axis=0,
            )
            step_bias = lr if self.bias_lr is None else self.bias_lr
        else:
            grad_bias = None
            step_bias = None
        if self.adam:
            self._adam_t += 1
            self._adam_step(self.theta, grad, self._m, self._v, lr)
            if grad_bias is not None:
                self._adam_step(self.theta_bias, grad_bias,
                                self._m_bias, self._v_bias, step_bias)
        else:
            self.theta.assign_add(tf.cast(lr, tf.float32) * grad)
            if grad_bias is not None:
                self.theta_bias.assign_add(tf.cast(step_bias, tf.float32) * grad_bias)

    def mode_bits(self):
        """Most probable individual: (D,) bool, bit = probs > 0.5."""
        return self.probs() > 0.5

    def mode_bias(self):
        """Current learned bias estimate for the mode individual: (D_b,) float32."""
        return self.theta_bias

    def diagnostics(self, layer_sizes=None):
        """Entropy/commitment stats of the current Bernoulli distribution."""
        return diagnostics_dict(self.probs(), self.theta, layer_sizes=layer_sizes)

    def diagnostics_packed(self, layer_sizes=None):
        """Packed diagnostics tensor (see ``bes.algs._common.packed_diagnostics``)."""
        return packed_diagnostics(
            self.probs(), self.theta, self.fitness, layer_sizes=layer_sizes)
