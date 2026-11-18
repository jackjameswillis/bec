"""Gaussian evolutionary strategy for binary individuals, sign quantization.

Search parameters are continuous latents ``theta in R^D`` — the latent-weight /
inertia interpretation of the surrogate parameters of binary networks, as
opposed to the Bernoulli-expectation interpretation optimised by
``bes.algs.es.SoftES``. Every generation draws isotropic Gaussian perturbations
``eps ~ N(0, sigma^2 I)`` with a shared fixed scale ``sigma``, forms the
perturbed parameters ``x_i = theta + eps_i`` and quantizes them with the sign
map: individual ``i`` is the binary network with weights ``sign(x_i)``. Fitness
is therefore always evaluated on a valid binary network, and the fitness
function is exactly the same black box as for the Bernoulli ES; what changes is
where the search distribution lives (continuous parameter space) and the
gradient estimator.

The initial search parameters are drawn from a zero-mean Gaussian
``N(0, init_std^2)`` so exact zeros (where the sign map is undefined) have
measure zero; bit ``j`` is quantized as ``x_j > 0``.

REINFORCE gradient of the expected fitness
``J(theta) = E_eps[f(sign(theta + eps))]``:

    grad_theta log N(x; theta, sigma^2 I) = (x - theta) / sigma^2 = eps / sigma,

so with shaped utilities ``u_i`` the population-summed update is

    theta <- theta + lr * sum_i u_i * eps_i / sigma

(no division by P — repo population-summed convention). The per-coordinate
score ``eps / sigma`` has unit variance, vs the bounded Bernoulli score
``x - s`` of SoftES, so learning-rate units are comparable but the tails are
heavier. Because the search parameters are plain continuous coordinates
updated by an additive gradient estimate, optimizer preprocessing (Adam) can
be applied exactly like in the straight-through-estimator pipeline (the
``adam`` option below ports SoftES's per-coordinate Adam rule).

The induced distribution over binary vectors is a product of Bernoullis with
rates ``p_j = P(sign(theta_j + eps) = 1) = Phi(theta_j / sigma)`` (erf), i.e.
the same Bernoulli family as SoftES under a monotone reparameterisation of the
logits; the two algorithms differ in parameterisation, score function, and
convergence dynamics (commitment now appears as ``|theta_j| >> sigma`` growth
instead of logit saturation).

Perturbation noise comes either from a pre-generated table of
theta-independent standard normals (``table_rows``; reusable at every theta
since ``eps`` does not depend on ``theta``, circular rows like SoftES) or from
a streaming sampler (``table_rows=None``). The REINFORCE score needs the float
perturbations again at update time, but the ``(P, D)`` noise tensor is never
materialised whole: ``sample()`` and ``update()`` replay the identical
row-chunked noise stream, re-created per generation from a splitmix64-mixed
seed (TF generators cannot snapshot/restore mid-stream, and raw consecutive
seeds produce overlapping philox streams — see ``_noise_seed``). Streaming and
table runs are statistically equivalent, not bit-identical.

With ``antithetic=True`` (default off) each generation consumes
``ceil(P / 2)`` noise rows and pairs each row ``eps`` with ``-eps`` (the pair
mates are ``sign(theta + eps)`` and ``sign(theta - eps)`` — unlike the
Bernoulli case at p = 0.5 these bits are NOT complements in general, the
variance reduction comes from the mirrored score ``-eps``). An odd population
appends one unpaired row.

Fitness shaping (``shaping``) follows SoftES: ``"none"`` = raw advantage
(unbiased), ``"standardize"`` = population z-score, ``"rank"`` = centered
average ranks in ``[-0.5, 0.5]``. With ``adam=True`` the population-summed
gradient goes through the same bias-corrected per-coordinate Adam rule as
SoftES. ``update(lr, grad_chunk=k)`` sub-chunks the population multiply-reduce
so the ``(rows, D)`` float temporaries stay bounded for large populations; it
does NOT change the noise stream (only the fp summation order).

Diagnostics mirror SoftES through the induced Bernoulli rates ``p =
Phi(theta / sigma)``: mean per-bit entropy, committed fraction (``p < 0.01``
or ``p > 0.99``), plus a Gaussian-specific ``flip_frac`` — the mean probability
that a perturbation flips the mode bit, ``E[P(sign(theta + eps) != sign(theta))]
= E[Phi(-|theta| / sigma)]``, the sigma-relative commitment measure.

Weights-only: like the locked Bernoulli-paper protocol there are no bias
coordinates; the fitness callable receives ``bias=None``.
"""

import tensorflow as tf

from bes.algs._common import (
    ESCommon, diagnostics_dict, packed_diagnostics, row_chunk_size,
)

_M64 = 0xFFFFFFFFFFFFFFFF
_SQRT2 = 1.4142135623730951


def _noise_seed(base_seed, generation):
    """63-bit splitmix64 mix of (base_seed, generation).

    Streaming mode re-creates the per-generation noise generator from this
    seed in both ``sample()`` and ``update()``. Raw consecutive seeds give
    overlapping philox streams (the seed becomes the counter), so generations
    are decorrelated through this hash before reaching ``from_seed``.
    """
    z = (int(base_seed) + (int(generation) + 1) * 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return ((z ^ (z >> 31)) & _M64) >> 1


class GaussianES(ESCommon):
    """Isotropic Gaussian search distribution over continuous latents with
    sign quantization; REINFORCE (score ``eps / sigma``) population update.
    """

    def __init__(self, num_bits, population_size, sigma=1.0, init_std=1.0,
                 table_rows=1024, antithetic=False, seed=None, shaping="none",
                 adam=False, beta1=0.9, beta2=0.999, eps=1e-8):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive (shared fixed perturbation scale)")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if int(num_bits) <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")
        if int(population_size) <= 0:
            raise ValueError(
                f"population_size must be positive, got {population_size}")
        if shaping not in ("none", "standardize", "rank"):
            raise ValueError(f"unknown fitness shaping: {shaping}")
        if table_rows is not None and int(table_rows) < 0:
            raise ValueError(f"table_rows must be non-negative, got {table_rows}")
        self.num_bits = int(num_bits)
        self.population_size = int(population_size)
        self.sigma = float(sigma)
        self.init_std = float(init_std)
        self.antithetic = bool(antithetic)
        self.shaping = str(shaping)
        self.adam = bool(adam)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._adam_t = 0
        self.table_rows = None if not table_rows else int(table_rows)
        self._base_seed = 0 if seed is None else int(seed)
        self._gen = tf.random.Generator.from_seed(self._base_seed)
        self.theta = tf.Variable(
            self._gen.normal([self.num_bits], stddev=self.init_std),
            trainable=False)
        self._table = (
            None if self.table_rows is None
            else self._gen.normal([self.table_rows, self.num_bits], stddev=1.0)
        )
        self._rows_used = 0
        self._noise_idx = None   # table mode: circular row indices of the current generation
        self._generation = 0
        # adaptive streaming row-chunk: bounds each (rows, D) float noise chunk
        # to ~1 GB/3 for wide nets (same rule as SoftES)
        self._stream_chunk = row_chunk_size(self.num_bits)
        self._m = tf.Variable(tf.zeros([self.num_bits], tf.float32), trainable=False)
        self._v = tf.Variable(tf.zeros([self.num_bits], tf.float32), trainable=False)
        self.population = None
        self.fitness = None

    def probs(self):
        """Induced per-bit Bernoulli rates: P(sign(theta_j + eps) = 1) = Phi(theta_j / sigma)."""
        return 0.5 * (1.0 + tf.math.erf(self.theta / (self.sigma * _SQRT2)))

    def _noise_shape(self):
        """(paired_rows, extra_rows) noise rows of the current generation.

        Each paired noise row generates a ``+eps`` / ``-eps`` individual pair,
        so ``P // 2`` rows cover ``2 * (P // 2)`` individuals and the odd
        tail row covers the remaining one (ceil(P/2) rows total, matching
        SoftES's antithetic consumption).
        """
        if self.antithetic:
            return self.population_size // 2, self.population_size % 2
        return self.population_size, 0

    def rows_per_generation(self):
        """Noise rows consumed per generation (antithetic pairs halve it)."""
        pair_rows, extra = self._noise_shape()
        return pair_rows + extra

    def _eps_chunks(self, pair_rows, extra):
        """Yield (eps_chunk, paired) rows of standard normals for the current
        generation, in the exact order both ``sample()`` and ``update()``
        replay.

        Table mode gathers circular blocks of the pre-generated table (the
        consumed row indices were stored by ``sample()``); streaming mode
        re-creates the per-generation generator from the mixed seed. Paired
        rows only occur under ``antithetic`` (each yields a ``+eps`` and a
        ``-eps`` population half); unpaired rows (non-antithetic, or the odd
        antithetic tail) yield a single ``+eps`` half.
        """
        if self._table is not None:
            idx = self._noise_idx
            pos = 0
            for i in range(0, pair_rows, self._stream_chunk):
                k = min(self._stream_chunk, pair_rows - i)
                yield tf.gather(self._table, idx[pos:pos + k]), self.antithetic
                pos += k
            if extra:
                yield tf.gather(self._table, idx[pos:pos + extra]), False
        else:
            gen = tf.random.Generator.from_seed(
                _noise_seed(self._base_seed, self._generation))
            for i in range(0, pair_rows, self._stream_chunk):
                k = min(self._stream_chunk, pair_rows - i)
                yield gen.normal([k, self.num_bits], stddev=1.0), self.antithetic
            if extra:
                yield gen.normal([extra, self.num_bits], stddev=1.0), False

    def sample(self):
        """Resample the population: (P, D) bool, bit = (theta + sigma*eps) > 0."""
        pair_rows, extra = self._noise_shape()
        self._generation += 1
        theta = self.theta[None, :]
        if self._table is not None:
            start = self._rows_used % self.table_rows
            self._rows_used += pair_rows + extra
            self._noise_idx = tf.range(
                start, start + pair_rows + extra) % self.table_rows
        parts = []
        for eps, paired in self._eps_chunks(pair_rows, extra):
            e = self.sigma * eps
            if paired:
                parts.append((theta + e) > 0.0)
                parts.append((theta - e) > 0.0)
            else:
                parts.append((theta + e) > 0.0)
        self.population = tf.concat(parts, axis=0) if len(parts) > 1 else parts[0]
        return self.population

    def evaluate(self, fitness_fn):
        self.fitness = fitness_fn(self.population, None)

    def _reduce(self, u, mat, grad_chunk):
        """sum_i u_i * mat_i over the rows of one noise chunk."""
        m = int(mat.shape[0])
        if grad_chunk is None or grad_chunk >= m:
            return tf.reduce_sum(u[:, None] * mat, axis=0)
        parts = []
        for i in range(0, m, grad_chunk):
            j = min(i + grad_chunk, m)
            parts.append(tf.reduce_sum(u[i:j, None] * mat[i:j], axis=0))
        return tf.add_n(parts)

    def update(self, lr, grad_chunk=None):
        """REINFORCE step: theta <- theta + lr * sum_i u_i * eps_i / sigma.

        Replays the current generation's noise stream chunk-by-chunk (see
        ``_eps_chunks``); the ``(P, D)`` perturbation tensor is never
        materialised whole. Must be called after ``sample()`` and
        ``evaluate()``.
        """
        u = self._utility()
        pair_rows, extra = self._noise_shape()
        parts = []
        pos = 0
        for eps, paired in self._eps_chunks(pair_rows, extra):
            k = int(eps.shape[0])
            if paired:
                # population order per chunk: +eps half then -eps half
                parts.append(self._reduce(u[pos:pos + k], eps, grad_chunk)
                             + self._reduce(u[pos + k:pos + 2 * k], -eps, grad_chunk))
                pos += 2 * k
            else:
                parts.append(self._reduce(u[pos:pos + k], eps, grad_chunk))
                pos += k
        grad = tf.add_n(parts) / self.sigma
        if self.adam:
            self._adam_t += 1
            self._adam_step(self.theta, grad, self._m, self._v, lr)
        else:
            self.theta.assign_add(tf.cast(lr, tf.float32) * grad)

    def _flip(self):
        """Mean perturbation flip probability ``E[Phi(-|theta| / sigma)]``."""
        return 0.5 * tf.math.erfc(tf.abs(self.theta) / (self.sigma * _SQRT2))

    def mode_bits(self):
        """Most probable individual under the induced Bernoulli rates: (D,) bool."""
        return self.theta > 0.0

    def diagnostics(self, layer_sizes=None):
        """Entropy/commitment/flip stats of the induced Bernoulli rates."""
        return diagnostics_dict(self.probs(), self.theta, extra=self._flip(),
                                layer_sizes=layer_sizes)

    def diagnostics_packed(self, layer_sizes=None):
        """Packed diagnostics tensor (see ``bes.algs._common.packed_diagnostics``)."""
        return packed_diagnostics(self.probs(), self.theta, self.fitness,
                                  extra=self._flip(), layer_sizes=layer_sizes)
