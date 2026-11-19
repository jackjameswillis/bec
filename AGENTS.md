# Agent Guide for `bes` — Binary Evolutionary Strategies

## Setup

- **Python:** use the ste venv `/data/ste/venv/bin/python` (Python 3.11, TF
  2.21.0, larq 0.14.0). No per-repo venv, no `requirements.txt`/`pyproject.toml`,
  no package manager, no Makefile, no CI or lint/typecheck config; tests run via
  `tests/run_all.py`.
- **GPU (RTX 3060 Ti):** the ste venv's `sitecustomize.py` ctypes-preloads the
  pip nvidia wheels, so plain `/data/ste/venv/bin/python …` gets GPU:0. The
  `experiments/tfgpu.sh` wrapper remains as a fallback for custom venvs. GPU
  reduction order is not bitwise deterministic vs CPU runs.
- No install step: scripts/tests add the repo root via `sys.path.insert(0, ...)`.
  Run as `cd /data/bes && /data/ste/venv/bin/python <script>`.
- Memory: wide-net runs need smaller eval chunks (`--chunk`) and population is
  bounded by the `(P, D)` float temporaries (~6 GB GPU: h1024×3 pop-100 OK at
  chunk 1024, pop-256 OOMs; h1024×1 pop-128/anti OK at chunk 512). There is no
  table-size guard; OOM is the failure mode.
- **Results:** only paper-used runs are retained, with regeneration notes in
  `results/README.md`. The full pre-prune archive is
  `/data/bes_results_archive_20261118.tar.zst` (checksum alongside). Detailed
  historical notes and recorded results live in `docs/experiments.md`.

## Commands

- Tests: `/data/ste/venv/bin/python tests/run_all.py` runs every
  `tests/test_*.py` sequentially in fresh processes (~100 s). Individual
  scripts: `test_entrypoints.py` (CLI smoke + history ordering),
  `test_resume.py`, `test_ste_bias.py`, `test_output_names.py`,
  `test_gradient.py`, `test_shaping.py`, `test_es_table.py`, `test_ges.py`,
  `test_mga.py`, `test_fitness.py`, `test_forward.py`.
- STE baseline: `cd /data/bes && TF_USE_LEGACY_KERAS=1 /data/ste/venv/bin/python experiments/run_ste.py [--epochs 25] [--batch-size 512] [--learning-rate 0.001] [--hidden 64] [--layers 1] [--scale none|fixed|he] [--eval-every N] [--optimizer sgd|adam] [--no-input-quant] [--use-bias] [--seed 42] [--out PATH]` (default stems include lr, epochs, batch size).
- BES: `cd /data/bes && /data/ste/venv/bin/python experiments/run_es.py [--population 100] [--lr 0.1] [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge] [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--table-rows 1024] [--antithetic] [--sigma 0.1] [--shaping none|standardize|rank] [--bias-lr LR] [--adam] [--natural] [--grad-chunk K] [--pop-chunk K] [--hidden 64] [--layers 1] [--scale fixed|he|none] [--activation sign|relu] [--chunk 8192] [--diagnostics] [--save-theta] [--out PATH]`.
- Gaussian ES: `cd /data/bes && /data/ste/venv/bin/python experiments/run_ges.py` with the same architecture/memory flags plus `--init-std 1.0` and `--sigma 1.0`; weights-only (fitness gets `bias=None`).
- Microbial GA: `cd /data/bes && /data/ste/venv/bin/python experiments/run_mga.py [--population 100] [--deme 2] [--crossrate 0.5] [--mutation-rate 0.001] [--bias-std 0.01] [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge] [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--scale fixed|he|none] [--activation sign|relu] [--hidden 64] [--layers 1] [--chunk 8192] [--pop-chunk 0] [--out PATH]`.
- MGA sweep: `cd /data/bes && /data/ste/venv/bin/python experiments/run_mga_sweep.py [--demelist ...] [--crossrates ...] [--seeds 0,1,2] [--population 256] [--fitness-batch 512] [--iterations 2000] [--mutation-rate 1e-4] [--bias-std 1.0] [--eval-every 10] [--eval-chunk 2048] [--scale fixed|he|none] [--activation sign|relu] [--resume] [--rerun-stale] [--out-dir results/mga_sweep_exact]`.
- MGA figures: `experiments/plot_mga_sweep.py [--in-dir …] [--out-dir …]`; collapse figure: `experiments/plot_cr05_collapse.py [--in-dir …] [--cr 0.5] [--out …]`.
- Paper sweep: `experiments/run_paper_sweep.py [--seeds 42,43,44] [--iterations 1000] [--eval-every 10] [--pops 128,256,512,1024] [--lr-scale 100] [--sweep-pop 0] [--depth-hidden 256] [--base-hidden 1024] [--depths 1,2,3] [--shapings none,standardize,rank] [--batches 256,1024,4096] [--arms pop,depth,shape,batch] [--out-dir results/paper_sweep] [--resume] [--rerun-stale] [--dry-run]`; lr = `--lr-scale`/population. Figures: `experiments/plot_paper_sweep.py`.
- Convergence figures: `experiments/plot_convergence.py [--in-dir results] [--out-dir figures/convergence] [--tag h256x2] [--pattern GLOB] [--iterations 2000] [--hist-cols 1,50,500,2000]`.
- Long-run/STE figures: `experiments/plot_long_ste.py [--in-dir results] [--out-dir figures/convergence] [--es-glob GLOB] [--ste-glob GLOB] [--iterations 5000]`.
- Theta figures: `experiments/plot_theta_expectations.py` and `plot_theta_neurons.py` (default npz is the retained seed-42 long-run archive).
- Rank plan: `PY=/path/to/python bash experiments/run_h256x2_rank_plan.sh` (defaults to `/data/ste/venv/bin/python`).

## Architecture

- TF-only repo (tf.keras + larq 0.14); no torch here. All algorithms maximise
  fitness; `bes/fitness.py` returns `-loss` for `kind="loss"` and raw accuracy
  for `kind="accuracy"`.
- Binary individual = flat bits over both weight matrices, sign-mapped
  `z = 2x - 1`, plus optional continuous biases. Default architecture constants:
  784 -> 64 -> 10 in `bes/mlp.py`; paper recipes override this (h256×2).
- `bes/mlp.py`: `num_bits`, `weight_shapes`, `bias_shapes`, `num_bias`,
  `split_weights`, `split_biases`, `quantise_input`, `population_metrics`
  (sample-chunked, default `chunk_size=8192`; `pop_chunk` for population-axis
  chunking), `net_probs` (alias `net_logits`). Scale modes: `fixed`
  (1/sqrt(in_dim)), `he` (sqrt(2/in_dim)), `none`. Activations: `sign` or `relu`.
  `quantise_input` is test-only; ES/MGA fitness sees raw normalised pixels.
- `bes/ste.py`: larq `QuantDense` with `ste_sign` quantizers and `weight_clip`;
  default no biases; `--use-bias` adds biases after the per-layer scale, matching
  the ES forward. First-layer input quantisation is on by default;
  `--no-input-quant` feeds raw pixels.
- `bes/data.py`: MNIST, normalised by train mean/std; validation is the last 10%
  of the training split (6000 samples); returns `(fit, val, test)`.
- `bes/algs/_common.py`: ES helpers — fitness shaping (rank uses argsort
  ordinal ranks, O(P) memory), Adam step, fitness stats, entropy/commitment
  diagnostics, packed diagnostics, and adaptive noise row-chunk size.
- `experiments/_common.py`: split loading, shape/fitness construction, metrics,
  snapshot selection, theta dumping, CLI validation, and the shared
  `run_search` loop. History is logged after each update.
- `experiments/_names.py`: default output-stem builders. `experiments/_resume.py`:
  cached sweep config validation.
- `bes/algs/es.py`: BES — Bernoulli logits over weight bits plus Gaussian bias
  coordinates. REINFORCE with mean-fitness baseline, population-summed update;
  options: shaping (`none`/`standardize`/`rank`), `sigma=0` weights-only,
  `bias_lr`, `adam`, `natural`, `antithetic`, table/streaming noise, and
  `--chunk`/`--grad-chunk`/`--pop-chunk` memory levers.
- `bes/algs/ges.py`: GaussianES — continuous latents, sign-quantized samples,
  fixed shared `sigma`, `init_std` Gaussian init; weights-only; `theta` is
  clipped to `[-1, 1]` at init and after each update; the perturbation stream is
  replayed chunk-by-chunk at update time. Table memory is `T*D*4` bytes; use
  `--table-rows 0` for wide nets.
- `bes/algs/mga.py`: MicrobialGA — persistent Bernoulli-0.5 population, ring
  tournament matching, uniform crossover + mutation; population must be even.
- `bes/fitness.py`: `BNNFitness` schedules `random` (default) and `cycle`
  (mgab protocol); loss kinds `ce` / `hinge`.
- Current paper recipe: h256×2, relu, `scale=he`, loss/CE, `sigma=0`, `rank`,
  `table_rows=0`, P=2048 with lr≈100/P, seed set 42–46; STE comparison uses
  h256×2, he, no-input-quant, Adam 1e-3.

## Gotchas

- larq import requires legacy Keras: run STE commands with
  `TF_USE_LEGACY_KERAS=1`; all other runners/tests work either way.
- Reproducibility: runners seed TF; BES/GaussianES/MGA own their generators.
  Table and streaming modes are statistically equivalent, not bit-identical.
  GPU and CPU reduction orders differ.
- `bes/mlp.py` `weight_shapes` chains hidden widths; `tests/test_forward.py`
  covers 3-hidden-layer parity.
- `run_paper_sweep.py` / `run_mga_sweep.py` resume only caches with matching
  configs; pass `--rerun-stale` to overwrite stale caches.
- `results/` and `figures/` are git-ignored except `results/README.md`; canonical
  paper figures are tracked under `submission/bernoulli-paper/`.

## Conventions

- All algorithms maximise fitness; code is authoritative over `DESIGN.md` prose.
- `__init__.py` files stay empty; imports use direct module paths.
- Credential/push policy: see `/data/AGENTS.md`; never push or force-push without
  explicit per-action confirmation.

## References

- Design: `DESIGN.md`; detailed history: `docs/experiments.md`; retained data:
  `results/README.md`.
- Literature: `papers/bnn_optimization_literature.md`.
- STE source of truth: `/data/ste/train_mnist_bnn.py`.
- Soft-ES reference: `/data/comb-exp/algs/es.py`; microbial-GA reference:
  `/data/mgab`.
