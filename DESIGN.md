# Design: `bes` — Binary Evolutionary Strategies

## Context

Paper: "Evolutionary Strategies for Binary Neural Network Optimization"
Target venue: EvoStar 2027 (EvoApplications track), EasyChair `conf=evostar2027`.

Two optimisation methods to compare on binary neural networks:

1. **Straight-Through Estimator (STE)** — from `/data/ste/train_mnist_bnn.py` (TF + Larq, `QuantDense` with `input_quantizer="ste_sign"`, `kernel_quantizer="ste_sign"`, `kernel_constraint="weight_clip"`; SGD lr 0.001, batch 512, 25 epochs, categorical cross-entropy, no momentum/label smoothing; mean/std normalisation; input binarised by default, with `--no-input-quant` for full-precision pixels).

2. **Soft Evolutionary Strategy** — Bernoulli-distributed binary weights via sigmoid expectation (`p(z=+1) = σ(θ)`), REINFORCE gradient of expected fitness, mean-fitness baseline, `denom = θ(1−θ)` (derivation in `/data/comb-exp/algs/es.py` and `/data/comb-exp/essays/es_derivation.md`; REINFORCE maths in `/data/reinforce_maths.tex`; note the `θ(1−θ)` denominator cancels exactly under the sigmoid reparameterisation used in `bes/algs/es.py` — see its docstring).

3. **Microbial Genetic Algorithm** — persistent population evolved by Harvey-style microbial tournaments (PyTorch reference `/data/mgab`), ported to TF as `bes/algs/mga.py`.

## Locked decisions

1. **Framework: TensorFlow + Larq for the whole repo.** `/data/bes` will be TF-only (tf.keras + larq). STE lives here. Soft ES lives here.

2. **Fitness:** both a loss and an accuracy shall be implemented and available. **Default fitness = training loss** (categorical cross-entropy, loss kind `"ce"`), to match the STE experiment exactly. Accuracy is the secondary signal. Loss-based fitness minimises a selectable loss kind via `BNNFitness(loss=...)`: `"ce"` (categorical cross-entropy) or `"hinge"` (squared hinge). Loss-based fitness is negated (fitness-maximisation convention); accuracy passes through raw.

3. **Individual: full binary weight matrix plus continuous layer biases.** Weights are binary sign vectors over both weight matrices; biases are continuous (hidden + output layers), zero-initialised and optimised by the same SoftES population as the weights (Gaussian mean + fixed perturbation scale). (Flagged discrepancy: `/data/ste/train_mnist_bnn.py` docstring says `784 -> 512 -> 10` but code uses `HIDDEN_UNITS = 64`. Design follows the code unless overridden.)

4. **Single repo: `/data/bes`.**

## Individual representation

- Hidden layer: binary weights of shape `(m, n) ∈ {-1, +1}^{m×n}` via sign, activation ReLU, plus optional continuous bias `(m,)`.
- Output layer: binary weights of shape `(r, m) ∈ {-1, +1}^{r×m}`, activation softmax, plus optional continuous bias `(r,)`.
- Biases: continuous, zero-initialised, optimised by the ES population alongside the weights (Gaussian mean, fixed perturbation scale); not part of the binary vector. The locked paper recipes use `sigma=0` (weights-only); the bias machinery remains available for probes.
- Flattened per individual into a single binary vector for the ES probability vector `θ ∈ (0,1)^D` (comb-exp derivation used bits `x ∈ {0,1}` with `p(x=1) = θ`; here we use sign values `z ∈ {±1}` with `p(z=+1) = σ(θ)`, the sign flip `z = 2x − 1`, expectation `E[z] = 2σ(θ) − 1`; implemented in `bes/algs/es.py`).

## Soft ES (BNN) — algorithm

- Sample population of P binary nets from `θ` via Bernoulli; continuous layer biases sampled from Gaussian means `θ_b` (per-individual).
- Batch-evaluate fitness (loss default).
- REINFORCE update (population-summed estimator, as implemented in `bes/algs/es.py`): `θ_sum ← θ_sum + α Σ_i (f_i − b)(x_ij − σ(θ_j))`, with `b` = mean fitness over the population, `p(z=+1) = σ(θ_j)`, and the closed-form Bernoulli logit-space score `x − σ(θ)` (equivalent to the per-sample REINFORCE form `α (f − b) ∇ log p` up to the population sum — the update scales the effective learning rate by P); biases use the Gaussian score `∇_{θ_b} log p(b) = (b − θ_b)/σ²`, population-summed the same way. Fitness shaping (SoftES `shaping`, added 2026-11-11): the advantage `(f_i − b)` may be replaced by a bounded utility — `none` (raw, unbiased, default), `standardize` (population z-score), or `rank` (centered average ranks in [−0.5, 0.5]; Salimans et al. 1703.03864) — bounding the step and decoupling it from the fitness scale (exploding-CE runaway guard) at the cost of estimator bias.

## Microbial GA (BNN) — algorithm

- Persistent population (uniform Bernoulli-0.5 weight bits, zero continuous biases) evolved by microbial tournaments (port of `/data/mgab`): per iteration a random perfect matching over ring demes (each unpaired individual picks a random unpaired partner in its forward window `{i+1 … i+deme_size−1} (mod P)`; `deme_size ≥ P` = fully mixed), higher fitness wins, the loser per pair is overwritten in place.
- Loser update: uniform crossover (copy winner's gene per weight bit / per bias coordinate with probability `crossrate`, default 0.5; 1 = full clone) then mutation — per-weight-bit flips with probability `mutation_rate` (default 1e-3) and Gaussian bias noise `N(0, bias_std)` (default 0.01). Winners untouched (best individual never destroyed).
- Fitness = same batched BNN fitness as the ES (loss default, `P/2` pairs per generation, `BNNFitness` random batch per evaluation — mgab `score_mode='batch'` protocol). Population must be even.
- Eval candidate = current fitness argmax (`best_bits`/`best_bias`), evaluated on val/test as the ES evaluates its mode net (locked defaults, 2025-11: population 100, deme 2).

## Population inference

- Implemented in `bes/mlp.py`: `population_metrics` evaluates all `P` individuals with batched eager `tf.einsum` forward passes over sign-mapped weights, chunked on the sample axis (`chunk_size`) and optionally on the population axis (`pop_chunk`). The STE baseline keeps its separate Larq `QuantDense` pipeline in `bes/ste.py`.

## STE baseline

- Reuse/adapt `/data/ste/train_mnist_bnn.py` logic: SGD(lr=0.001), batch 512, epochs 25, ce loss (or squared hinge, loss kind `"hinge"`), momentum 0, label smoothing 0, `ste_sign` quantizers, `weight_clip`, momentum 0, normalise. The first layer binarises inputs by default (as in the original script); `--no-input-quant` feeds full-precision pixels. Biases are off by default (`--use-bias` to enable).

## Experiments

- MNIST. Soft ES vs STE. Claim: ES reaches performance comparable to STE on binary BNN optimisation.
- **Deme-size × crossrate sweep (MGA, "Arc 1" reproduction of `/data/mgab/pilot_sweep.py`):** MNIST, pop 256, fitness batch 512 (cycling mini-batch protocol), mutation rate 1e-5, bias std 1e-5, 1000 iterations, 3 seeds; grid = ring deme sizes {2,4,8,16,32,64,128,256} × crossrates {0.5, 1.0}. Hypothesis under test (mgab): restrictive ring geography improves accuracy and stability (fully-mixed matching = worst; deme-2 the most stable). Runner `experiments/run_mga_sweep.py`, figures `experiments/plot_mga_sweep.py` (trace + peak/collapse heatmap). Deviations from the mgab data are documented in the sweep-script docstring; mgab's recorded grid itself was a patchwork of iteration counts/mutation settings across crossrate cells.
- **Forward-mode axis (`--scale fixed|he|none`):** bes default = sign hidden activations with fixed 1/√(in_dim) pre-activation scaling; `he` = √(2/in_dim) per layer (He-style: with binary ±1 weights and ReLU hidden units the variance-stable scale is √2× the sign-net one — `E[x²]₊₁ = c²·fan_in·E[x²]/2` exactly, since symmetric ±1 weights zero out the mean term and make `E[relu(s)²] = E[s²]/2` exact); `none` = unscaled logits for mgab `r='linear'` parity. The recorded mgab data used ReLU hidden activations with no scaling — not bit-comparable to either mode; flagged in the paper comparison. First cr=1.0 sweep ran with `scale="fixed"` (partial archive in `results/mga_sweep_fixed_scale/`, demes 2–8 complete); re-run in progress with `scale="none"`.

## Status

All planned components are implemented: the STE baseline (`bes/ste.py`), the
SoftES / GaussianES search distributions (`bes/algs/es.py`, `bes/algs/ges.py`),
the microbial GA (`bes/algs/mga.py`), the batched BNN core (`bes/mlp.py`), and
the experiment runners/plotters under `experiments/`. Current recipes, sweeps,
and recorded results are maintained in `AGENTS.md`; use that file as the
operational reference rather than this design note.
