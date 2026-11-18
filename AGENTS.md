# Agent Guide for `bes` — Binary Evolutionary Strategies

## Setup

- **Python:** use the ste venv `/data/ste/venv/bin/python` (Python 3.11, TF
  2.21.0, larq 0.14.0). No per-repo venv, no `requirements.txt`/`pyproject.toml`,
  no package manager, no Makefile, no CI, no lint/typecheck/test-runner config.
- **GPU (RTX 3060 Ti):** TF sees it only via `experiments/tfgpu.sh …`
  (wraps the run with `LD_LIBRARY_PATH` pointing at the pip nvidia wheels in
  `site-packages/nvidia/*/lib` — TF never probes that path itself). CPU runs
  keep working unmodified; GPU gives ~4x on sweeps (fitness call 595→155 ms,
  full-val pop-256 eval 2.15 s→0.16 s). Note: GPU reduction order is not
  bitwise deterministic vs CPU runs.
  - **Automatic:** the ste venv's `site-packages/sitecustomize.py` now ctypes-
    preloads the nvidia wheels (no-op when no GPU), so plain
    `/data/ste/venv/bin/python …` scripts also get GPU:0 out of the box. The
    `tfgpu.sh` wrapper is retained as a fallback (e.g. custom venvs).
- No install step: scripts/tests add the repo root via `sys.path.insert(0, ...)`.
  Run as `cd /data/bes && /data/ste/venv/bin/python <script>`.
- GPU is the default path (see above); CPU-only fallback works (8 cores).
  Memory: wide-net runs need smaller eval chunks (`--chunk`) and population is
  bounded by the (P, D) float temporaries (~6 GB GPU: h1024×3 pop-100 OK at
  chunk 1024, pop-256 OOMs; h1024×1 pop-128/anti OK at chunk 512).

## Commands

- Tests (no pytest installed; plain scripts with `main()` + `if __name__ == "__main__"`):
  - `/data/ste/venv/bin/python tests/run_all.py` — run every `tests/test_*.py` script sequentially in fresh processes and summarise (OK, ~94 s)
  - `/data/ste/venv/bin/python tests/test_entrypoints.py` — `--help` smoke checks for runner/plot CLIs plus tiny ES/GES/MGA runs and history-ordering invariant (OK)
  - `/data/ste/venv/bin/python tests/test_resume.py` — sweep resume-cache config matching (OK)
  - `/data/ste/venv/bin/python tests/test_ste_bias.py` — STE `--use-bias` forward parity vs the ES model across scale modes (OK; sets legacy Keras internally)
  - `/data/ste/venv/bin/python tests/test_output_names.py` — default ES/GES/STE output-stem compatibility (OK)
  - `/data/ste/venv/bin/python tests/test_gradient.py` — REINFORCE gradient vs exact enumeration (OK)
  - `/data/ste/venv/bin/python tests/test_shaping.py` — SoftES fitness shaping: none/standardize/rank utilities + update parity; argsort rank matches the old `(P, P)` reference across sizes/ties (OK)
  - `/data/ste/venv/bin/python tests/test_es_table.py` — SoftES perturbation-table sampler (OK)
  - `/data/ste/venv/bin/python tests/test_ges.py` — GaussianES determinism/table-wraparound, known-noise numpy parity, streaming re-draw consistency, linear-fitness score unbiasedness vs the closed-form Phi gradient, SoftES shaping parity, grad-chunk parity, theta=0 diagnostics (OK)
  - `/data/ste/venv/bin/python tests/test_mga.py` — MicrobialGA pairing/tournament operators (OK)
  - `/data/ste/venv/bin/python tests/test_fitness.py` — BNNFitness batch schedules: cycle (mgab protocol) + random (OK)
  - `/data/ste/venv/bin/python tests/test_forward.py` — BNN forward/loss parity vs hand-rolled reference, fixed/he/none scale modes (pure TF, never imports larq; OK)
- STE baseline: `cd /data/bes && TF_USE_LEGACY_KERAS=1 /data/ste/venv/bin/python experiments/run_ste.py [--epochs 25] [--batch-size 512] [--learning-rate 0.001] [--hidden 64] [--layers 1] [--scale none|fixed|he] [--eval-every N] [--optimizer sgd|adam] [--no-input-quant] [--use-bias] [--seed 42] [--out PATH]` (default result stems include lr, epochs, and batch size)
- Soft ES: `cd /data/bes && /data/ste/venv/bin/python experiments/run_es.py [--population 100] [--lr 0.1] [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge] [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--table-rows 1024] [--antithetic] [--sigma 0.1] [--shaping none|standardize|rank] [--bias-lr LR] [--adam] [--grad-chunk K] [--natural] [--out PATH]` plus `--hidden 64` (width), `--layers 1` (hidden depth), `--scale fixed|he|none`, `--activation sign|relu`, `--chunk 8192` (sample chunk; GPU memory lever for wide nets — h1024 pop-100 needs `--chunk 1024` or 512)
- Gaussian ES (sign-quantized, `submission/gaussian-paper`): `cd /data/bes && /data/ste/venv/bin/python experiments/run_ges.py [--population 100] [--lr 0.1] [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge] [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--table-rows 1024] [--antithetic] [--sigma 1.0] [--init-std 1.0] [--shaping none|standardize|rank] [--adam] [--grad-chunk K] [--out PATH]` plus `--hidden/--layers/--scale/--activation/--chunk/--pop-chunk/--diagnostics/--save-theta` (same semantics as run_es.py; no biases — weights-only, fitness gets `bias=None`)
- Microbial GA: `cd /data/bes && /data/ste/venv/bin/python experiments/run_mga.py [--population 100] [--deme 2] [--crossrate 0.5] [--mutation-rate 0.001] [--bias-std 0.01] [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge] [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--scale fixed|he|none] [--out PATH]` plus `--hidden 64` (width), `--layers 1` (hidden depth), `--activation sign|relu`, `--chunk 8192`, `--pop-chunk 0`
- MGA deme sweep (runner defaults reproduce mgab's exact recipe; for the Arc-1 archive use `--iterations 1000 --mutation-rate 1e-5 --bias-std 1e-5 --out-dir results/mga_sweep`): `cd /data/bes && /data/ste/venv/bin/python experiments/run_mga_sweep.py [--demelist 2,4,8,16,32,64,128,256] [--crossrates 0.5,1.0] [--seeds 0,1,2] [--population 256] [--fitness-batch 512] [--iterations 2000] [--mutation-rate 1e-4] [--bias-std 1.0] [--eval-every 10] [--eval-chunk 2048] [--scale fixed|he|none] [--activation sign|relu] [--resume] [--rerun-stale] [--out-dir results/mga_sweep_exact]`
- MGA sweep figures: `cd /data/bes && /data/ste/venv/bin/python experiments/plot_mga_sweep.py [--in-dir results/mga_sweep_exact] [--out-dir figures/mga_sweep]` (matplotlib installed in the ste venv)
- MGA collapse figure (2-row: mean acc / mean loss-log traces + peak→final per deme): `cd /data/bes && /data/ste/venv/bin/python experiments/plot_cr05_collapse.py [--in-dir results/mga_sweep_exact] [--cr 0.5] [--out figures/cr05_collapse.pdf]`

- Paper sweep (population/depth/shaping/batch, one factor at a time around the
  base recipe h1024x1 relu he CE sigma=0 standardize b4096): `cd /data/bes &&
  /data/ste/venv/bin/python experiments/run_paper_sweep.py [--seeds 42,43,44]
  [--iterations 1000] [--eval-every 10] [--pops 128,256,512,1024] [--lr-scale
  100.0] [--sweep-pop 0] [--depth-hidden 256] [--depths 1,2,3] [--shapings
  none,standardize,rank] [--batches 256,1024,4096] [--arms
  pop,depth,shape,batch] [--out-dir results/paper_sweep] [--resume] [--rerun-stale] [--dry-run]`.
  lr = `--lr-scale`/population (step-size rule for the population-summed
  estimator; P=100↔lr 1, P=1024↔lr 0.0977). Each run = one subprocess of
  `run_es.py` (fresh TF process; recorded-working GPU levers per arch).
  13 cells × 3 seeds = 39 runs, 11 unique configs (the p1024 base cell
  dedupes with shaping-standardize and batch-4096); per-seed JSONs + stdout
  logs under `results/paper_sweep/runs/`, `summary.json` with final/peak
  val, test, wall time per cell. ~7 h GPU total.
- Paper sweep v2 (pop/depth/batch at h256, 2000 it, `--base-hidden 256` flag
  sets the pop/shape/batch arm width; runs go to a fresh out-dir because run
  tags don't encode iterations): `cd /data/bes && /data/ste/venv/bin/python
  experiments/run_paper_sweep.py --seeds 42,43,44 --iterations 2000 --arms
  pop,depth,batch --base-hidden 256 --out-dir results/paper_sweep_h256_it2000
  --resume [--rerun-stale]`; 10 cells × 3 seeds = 30 runs, 8 unique configs (p1024 = h256x1
  depth cell = b4096 batch cell dedupe); ~5 h GPU.
- Paper sweep figures: `cd /data/bes && /data/ste/venv/bin/python
  experiments/plot_paper_sweep.py [--in-dir results/paper_sweep] [--out-dir
  figures/paper_sweep]` — PDF+PNG pairs per arm (`fig_pop_scaling` (acc vs P
  log-x + wall-vs-P linear-cost guide), `fig_depth`, `fig_shaping`,
  `fig_batch`) + `fig_overview` 2x2; traces averaged over seeds with std
  bands, bars = final test acc mean±std with value labels; bar-panel y-lims
  auto-fit so sub-floor cells (sh none 0.9245) stay visible; arms missing
  from summary.json are skipped; `--ref-dir results/paper_sweep` overlays
  the h1024x1 width reference on the v2 depth figure.

## Architecture

- TF-only repo (tf.keras + larq 0.14); no torch here (lives in `/data/comb-exp`).
- All algorithms maximise fitness (comb-exp convention); `bes/fitness.py` returns
  `-loss` for `kind="loss"` and passes accuracy through raw (higher = better).
- Default fitness = mean categorical cross-entropy (negated); accuracy is the
  secondary signal.
- Binary individual = flat bits `{0,1}^D` over both weight matrices, sign-mapped
  `z = 2x - 1`, **plus continuous layer biases** (hidden + output) optimised by the
  ES in the same run (Gaussian mean + fixed perturbation scale). Architecture
  784 -> 64 -> 10 (`bes/mlp.py` constants; the `/data/ste` docstring's "512" is stale — trust the code).
- **Small baseline** (ES probe target, 2026-11-11): 784 -> 64 -> 10, one hidden
  layer, `activation="relu"`, `scale="none"` — the exact architecture of the
  `results/mga_sweep_exact/` collapse sweep. ES recipe on it that learns then
  shows the CE-improves/accuracy-decays arc: `run_es.py --hidden 64 --layers 1
  --activation relu --scale none --fitness loss --loss ce --lr 0.01 --sigma 0`
  (peak val ≈ 0.58 around it 100, then decays). Sigma ladder at lr 0.01:
  σ=0 → peak 0.582, σ=1 → 0.554, σ=10 → 0.649 (best; mild decay at 300 it),
  σ=20 → 0.597, σ=0.1/100 → noise-dominated chance, σ≤0.01 → 1/σ² bias-score
  explosion + immediate freeze at the huge-bias degenerate state. Note: this is
  a relu net, so `scale="he"` (sqrt(2/fan_in), added 2026-11-11) is the
  variance-stable scale for it — the recorded ladder used `scale="none"`.
  ES re-probe at `--scale he` (σ=0, pop 100, seed 42, 300 it): lr 0.01 crawls
  (val ≈ 0.15 at it 300, still climbing — he puts logit std at ≈8 vs ≈158 for
  none, so per-bit-flip CE swings and hence REINFORCE steps shrink ~proportionally);
  lr 0.1 climbs monotonically to val 0.612 / test 0.596 at it 300, still rising,
  no accuracy-decay arc through the same window (recorded scale-none σ=0:
  peak 0.582 @ ~it 100 then decays); lr 1 jumps to val ≈0.80 by it 100 and
  plateaus ≈0.85 val / 0.823 test at it 300 (peak ≈0.852 @ it 250, mild wobble
  late, no decay arc) — the best ES result recorded on this baseline by a wide
  margin (recorded none-scale best: 0.649). Ladder so far (σ=0, 300 it):
  lr 0.01 → 0.15 crawl, lr 0.1 → 0.61 climbing, lr 1 → 0.85 plateau. JSONs:
  `results/es_loss_ce_h64x1_he_relu_p100_lr0.01_seed42_t1024_20261111_175004.json`,
  `results/es_loss_ce_h64x1_he_relu_p100_lr0.1_seed42.json`,
  `results/es_loss_ce_h64x1_he_relu_p100_lr1_seed42.json`.
  Bias-σ probes at `--scale he` (2026-11-11): at lr 1 EVERY nonzero σ collapses
  under the REINFORCE bias score — σ=0.1 (raw) instant 1/σ² explosion then
  freeze, σ=1 (raw) chaotic divergence (CE 40-340), σ∈{0.1,1,10} with
  `--shaping rank|standardize` also collapse (shaping bounds the advantage but
  not the score's 1/σ² noise; standardize ≥ rank in damage — heavier-tailed
  utilities; JSONs `..._lr1.0_seed42_t1024_sh{rank,standardize}_*.json` and
  `..._lr1.0_seed42_t1024_20261111_210412.json`). At lr 0.1, σ=1 raw → 0.476
  (< 0.612 at σ=0): bias learning hurts at every probed (lr, σ>0). Stable
  recipe = `--sigma 0` (biases frozen at 0).
  Antithetic at lr 1 σ=0 (`--antithetic --sigma 0`, 300 it): val 0.8725 /
  test 0.8593, peak 0.8728 @ it 290, smooth monotone climb
  (`results/es_loss_ce_h64x1_he_relu_p100_lr1.0_seed42_t1024_anti_20261111_210701.json`),
  vs non-antithetic 0.849/0.823 with late wobble; mirrored rows halve the
  weight-gradient noise at p=0.5.
  Antithetic + `--shaping rank` at lr 1 σ=0 (300 it): val 0.9310 / test 0.9155,
  peak = final (still climbing at 300), monotone with zero wobble — BEST ES
  RESULT ON RECORD
  (`results/es_loss_ce_h64x1_he_relu_p100_lr1.0_seed42_t1024_anti_shrank_20261111_210932.json`).
  Rank shaping pays because raw advantages shrink with the CE scale (2.3 → 0.31
  over the run) so the raw step decays late; centered ranks keep the advantage
  scale constant (it 60: 0.8752 vs 0.7750 raw-advantage). Still climbing at 300.
  Same recipe at `--hidden 1024 --table-rows 0 --chunk 512` (h1024 = 813k bits;
  the 1024-row table OOMs 6 GB VRAM at init — 512 rows also fits but the user
  prefers no table): val 0.9360 / test 0.9274, peak 0.9375 @ it 280, mild late
  plateau — +0.5pp val / +1.2pp test for 13x params vs h64
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_seed42_t0_anti_shrank_20261111_211257.json`).
  h1024 with σ=1 in the same recipe collapses to chance too (test 0.133,
  CE stuck at −9 to −13) — width-independent failure, confirming `--sigma 0`.
  h1024 standardize twin (`--shaping standardize`, σ=0, anti, 300 it): val
  0.9425 / test 0.9296, peak 0.9432 @ it 280 — edges out rank (0.9360/0.9274):
  heavier-tailed z-score utilities pay for weights-only search (no bias-noise
  channel to blow up; reversal vs the σ>0 collapse runs where standardize
  damaged more than rank)
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_seed42_t0_anti_shstandardize_20261111_211753.json`).
  Non-antithetic standardize twin (h1024, σ=0, lr 1, 300 it): val 0.9420 /
  test 0.9289, peak 0.9428 @ it 280 — indistinguishable from the antithetic
  twin (0.9425/0.9296); antithetic's variance reduction only pays with raw
  advantages (h64 anti-none 0.8725 vs none-none 0.849), vanishing once shaping
  normalizes the advantage scale
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_seed42_t0_shstandardize_20261111_213743.json`).
  lr 0.1 non-anti standardize twin (h1024, σ=0, 300 it): val 0.9138 / test
  0.9041, peak 0.9162 @ it 290, still climbing — smooth but ~3-4x slower than
  lr 1 (it-300 trace ≈ lr 1's it-100); lr 1 remains the sweet spot
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr0.1_seed42_t0_shstandardize_20261111_214058.json`).
  lr 0.1 at 1000 it (same recipe): val 0.9498 / test 0.9403, peak 0.9508 @
  it 980, still gently rising — BEST RESULT ON RECORD; passes lr 1's 300-it
  mark (~0.942) around it 550-600 and keeps climbing
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr0.1_seed42_t0_shstandardize_20261111_214649.json`).
  lr 1 at 1000 it (same recipe): val 0.9577 / test 0.9485, peak 0.9585 @ it
  990, still climbing — the "plateau" at 300 it was just truncation; lr 1
  stays stable under standardize shaping and out-climbs lr 0.1 at equal budget
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_seed42_t0_shstandardize_20261111_215315.json`).
  Best recipe at `--layers 2` (h1024×2 = 1.86M bits, σ=0, lr 1, standardize,
  1000 it): val 0.9625 / test 0.9530, peak 0.9628 @ it 960 — NEW BEST; +0.5pp
  val / +0.45pp test over h1024×1, ~3.6pp over the untuned STE baseline
  (`results/es_loss_ce_h1024x2_he_relu_p100_lr1.0_seed42_t0_shstandardize_20261112_091805.json`).
  Same recipe at `--layers 3` (h1024×3 = 2.91M bits, chunk 256, 1000 it):
  val 0.9643 / test 0.9511, peak 0.9647 @ it 990 — +0.2pp val but −0.2pp test
  vs ×2 at 1.6x the compute: depth returns flat, h1024×2 stays the sweet spot
  (`results/es_loss_ce_h1024x3_he_relu_p100_lr1.0_seed42_t0_shstandardize_20261112_092825.json`).
  `--bias-lr` sweep at σ=1 (h1024, standardize, lr 1, non-anti, 300 it,
  2026-11-11): bias-lr 0.001 → 0.3975/0.3718 (slow smooth climb), 0.01 →
  0.5785/0.5618 (learns), 0.1 → collapsed 0.1452/0.1355; bias-lr 1.0 = the
  known collapse. First stable nonzero-σ runs, but the best is 36pp below the
  σ=0 reference (0.9420/0.9289) at equal budget — narrow usable band
  (≤0.01), SNR ∝ σ·√P verdict holding
  (JSONs `..._lr1.0_bl{0.001,0.01,0.1}_seed42_t0_shstandardize_*.json`).
  σ=0.1 at the same protocol (bias-lr 0.001): val 0.9050 / test 0.8883,
  still climbing at 300 — BEST nonzero-σ result by far (+33pp over the
  σ=1/bl-0.01 point at equal noise floor): σ's smoothing role dominates its
  SNR role; small perturbations keep individual fitness informative about
  the mean net
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_bl0.001_seed42_t0_shstandardize_20261111_221638.json`).
  σ=0.01 at the same protocol: chaotic collapse 0.1898/0.1904 (fitness mean
  bounces −2.7…−4.7; noise floor 1.0/coord/gen) — the σ-ladder for bias
  learning is an inverted-U: 0.01 chaos (1/σ² noise) → 0.1 sweet spot
  (0.9050) → 1 over-smoothing (0.40); two distinct failure modes bracket
  the peak
  (`results/es_loss_ce_h1024x1_he_relu_p100_lr1.0_bl0.001_seed42_t0_shstandardize_20261111_221949.json`).
- `bes/mlp.py`: `num_bits`, `weight_shapes`, `bias_shapes`, `num_bias`, `split_weights`,
  `split_biases`, `quantise_input`, `population_metrics` (sample-chunked, default
  `chunk_size` 8192; `pop_chunk=` population-axis chunking, row-exact), `net_logits`.
  NOTE: `quantise_input` is a helper only used by `tests/test_gradient.py` — the
  ES/MGA fitness and mode-net evals feed the RAW normalised pixels to the forward
  (full-precision input × ±1 weights); only the larq STE pipeline binarises inputs
  (`ste_sign` at the first QuantDense, `--no-input-quant` to disable). Pre-activation scale modes: `scale="fixed"`
  (default, 1/sqrt(in_dim) per layer — the Sari et al. 1909.09139 BN-replacement
  scale, correct for sign units), `scale="he"` (sqrt(2/in_dim) per layer —
  He-style variance-stable scale for binary weights with relu hidden units:
  `E[x²]₊₁ = c²·fan_in·E[x²]/2` exact, so the fixed point is √2× the sign-net
  scale) or `scale="none"` (unscaled logits, mgab pilot `r='linear'` parity;
  `BNNFitness(scale=...)` threads it through). Hidden
  activation modes: `activation="sign"` (default, binary hidden units) or
  `"relu"` (mgab-style dense ReLU); `BNNFitness(activation=...)` threads it through.
- `bes/ste.py`: larq `QuantDense` with `input_quantizer="ste_sign"`,
  `kernel_quantizer="ste_sign"`, `kernel_constraint="weight_clip"`; SGD lr 1e-3,
  momentum 0, categorical CE, no label smoothing, batch 512, 25 epochs.
  `use_bias=False` default; `run_ste.py --use-bias` adds biases after each
  per-layer pre-activation scale, matching the ES forward (`scale="none"`
  reduces to the original script's bias placement).
  `--hidden` flag (added 2026-11-11) sets the hidden width. h1024 runs
  (2026-11-11): no-bias val 0.9267 / test 0.9140, with-bias val 0.9240 /
  test 0.9140 — (a) the ES recipe (standardize, lr 1, 1000 it: 0.9577/0.9485)
  BEATS this untuned vanilla STE recipe by ~3pp on the same architecture
  (STE recipe inherited from h64, never tuned for h1024 — Adam probe
  warranted before claiming ES > STE), and (b) learned biases add nothing
  (noise-level difference), closing the bias-learning thread from an
  independent pipeline. Note huge CE (train 7.9 / val 10.3) despite 0.93
  acc — weight_clip drives extreme logit magnitudes (overconfident softmax).
  `--scale fixed|he|none` (added 2026-11-12; default none = original recipe):
  Lambda scale layers after each hidden QuantDense + pre-softmax output scale,
  mirroring `bes/mlp._scale` (relu homogeneity makes post-relu scaling
  equivalent). ×2 without scale collapses to chance (val 0.1208 / test 0.1157,
  saturated softmax — activations compound: hidden pre-act std 28 → 630);
  with `--scale he` ×2 trains but is undertrained at SGD 1e-3 × 25 epochs:
  val 0.6052 / test 0.5850, CE sane (1.44) — properly scaled gradients are
  ~1e4x smaller than the unscaled run's, so the h64-tuned lr is now too small
  (`results/ste_h1024x2_he_seed42_20261112_093703.json`).
  `--eval-every N` (added 2026-11-12): `EvalCallback` full-val evals every N
  training batches, printed ES-style and saved to `record["evals"]` (0 = off).
  The ×2-he trace over 2650 steps is a clean monotone climb with no plateau
  (0.09 → 0.34 @ 1050 → 0.49 @ 1850 → 0.605 @ 2650, ~+1pp/epoch at cutoff) —
  the STE baseline is strictly undertrained at the h64 recipe; more epochs /
  higher lr / Adam all on the table.
  100 epochs at lr 1e-3 (same recipe): val 0.8380 / test 0.8135, peak 0.8397
  @ step 10500, still climbing (~+0.5-1pp per 1000 steps late, decelerating);
  train acc only ~0.81 — the STE net UNDERFITS at this depth (sign-quantized
  + weight_clip + SGD momentum 0); 12pp below ES h1024×2 at 4x its
  iteration count
  (`results/ste_h1024x2_he_seed42_20261112_094608.json`).
  lr 0.01 at 100 epochs (same recipe): val 0.9233 / test 0.9080, peak 0.9260
  @ step 9850 — gap to ES h1024×2 shrinks to ~4pp (from 12pp at lr 1e-3);
  0.837 val after 1 epoch, then hard deceleration (+0.9pp over the last 4.5k
  steps; extrapolated plateau ≈ 0.93-0.94), CE healthy (0.27)
  (`results/ste_h1024x2_he_seed42_20261112_094906.json`).
  `--optimizer adam` (added 2026-11-12): Adam 1e-3 at 100 epochs (h1024×2,
  he): val 0.9822 / test 0.9812, peak 0.9842 @ step 3000, CE 0.075 — FAIR
  CEILING FOR THE ARCHITECTURE; converged in ~1 epoch (0.938 val @ step 50,
  0.977 @ 1050) then flat-wobbled. Verdict: tuned STE beats the best ES
  recipe by ~2.8pp test at ~10x less compute (Adam step ≈ 1 net-backward vs
  ES iteration ≈ 100 net-forwards); the earlier "ES beats STE" finding was
  an artifact of the untuned SGD-1e-3 recipe
  (`results/ste_h1024x2_he_adam_seed42_20261112_100502.json`).
  Adam-STE at `--layers 3` (same recipe): val 0.9847 / test 0.9831, peak
  0.9853 @ step 8800, CE 0.087 — depth pays +0.25pp val / +0.19pp test under
  the tuned STE recipe (unlike ES where ×3 was flat); NEW STE CEILING
  (`results/ste_h1024x3_he_adam_seed42_20261112_113621.json`).
  `--no-input-quant` (added 2026-11-12, `build_model(input_quant=False)`):
  first layer takes the raw normalised pixels (±1 weights × full-precision
  input instead of ±1 × ±1) — strictly more expressive first layer.
  Adam-STE h1024×3 he noiq: val 0.9883 / test 0.9864, peak 0.9890 @ step
  9700 (+0.36pp val / +0.33pp test over the input-quantized twin, faster
  early convergence 0.9605 vs 0.9423 @ step 50) — input binarisation costs
  ~0.3pp at this scale. CORRECTED ARCHITECTURE NOTE (2026-11-12): the ES
  pipeline NEVER binarised inputs (`quantise_input` is only called by
  tests/test_gradient.py) — the ES-matched function class was always the
  no-input-quant variant, so the like-for-like ceiling is 0.9883/0.9864 and
  the input-quantized STE was the restricted one
  (`results/ste_h1024x3_he_adam_noiq_seed42_20261112_114109.json`).
- `bes/data.py`: MNIST; full-precision input normalised by train mean/std (+1e-7);
  validation split = last 10% (6000 samples); returns (fit, val, test).
- `bes/algs/_common.py`: shared ES helpers — fitness shaping (`_utility`; rank
  uses an argsort average-rank implementation with O(P) memory), bias-corrected
  Adam step, fitness stats, binary entropy/commitment diagnostics
  and the packed diagnostics tensor, plus the adaptive noise row-chunk size.
- `experiments/_common.py`: shared runner plumbing — split loading, shape/fitness
  construction, per-individual and final split metrics, snapshot-step selection,
  theta dumping, and common CLI validation.
- `experiments/_names.py`: shared default output-stem builders for `run_es.py`,
  `run_ges.py`, and `run_ste.py`.
- `bes/algs/es.py`: SoftES — Bernoulli logit `p(z=+1) = sigmoid(theta)` over weight
  bits; continuous bias coordinates as Gaussian means `theta_bias` (fixed perturbation
  scale `sigma`; `sigma=0` disables bias learning — biases frozen at 0, update skips
  the bias gradient). REINFORCE with mean-fitness baseline, population-summed estimator
  (  Bernoulli score `(z - s)/2` for weights, Gaussian score `(b - theta_bias)/sigma^2`
  for biases; `bias_lr` decouples the bias step from the shared lr — score untouched,
  added 2026-11-11). Per-coordinate Adam preprocessing (`adam=True`, `run_es.py
  --adam`; added 2026-11-12): population-summed gradients go through a manual
  bias-corrected Adam rule (m/v buffers, shared step counter, eps 1e-8) instead of
  raw addition — steps bounded ~lr per coordinate, lr units shift accordingly
  (sweep 1e-4…1e-1). Raw path (default) bit-identical to before.
  Population-chunked update reduction (`update(lr, grad_chunk=k)`,
  `run_es.py --grad-chunk k`, 0 = full shot; added 2026-11-12): splits the
  population REINFORCE sum into k-row chunks (weight-bit cast included) so
  the (P, D) float temporaries stay bounded for large populations — pop-1000
  at h1024 was OOM-impossible before. Only changes fp summation order;
  parity-tested vs full shot. Three memory levers now: `--chunk` (batch axis
  of fitness), `--grad-chunk` (population axis of the REINFORCE reduction),
  `--pop-chunk` (population axis of fitness eval, row-exact; threaded through
  `BNNFitness(pop_chunk=...)`). Natural gradient (`natural=True`,
  `run_es.py --natural`; added 2026-11-12): divides the weight-bit gradient
  by the eps-damped diagonal Bernoulli Fisher `s(1-s)` — NES/Amari
  natural-gradient step; at p=0.5 a constant 4x lr rescale, diverges from
  raw as bits commit. The logit parameterisation and the Gaussian bias
  score `(b - theta_bias)/sigma^2` are already natural-gradient equivalent
  by construction.
  Natural-gradient probe (h1024×1, standardize, σ=0, pop-100, 300 it, seed
  42, 2026-11-12): nat lr 0.25 (step-matched: 4×grad × 0.25 = raw lr 1 at
  p=0.5) → val 0.9217 / test 0.9126; nat lr 1.0 → 0.9080/0.8936 — both
  BEHIND raw lr 1 (0.9420/0.9289) at every eval on identical population
  draws (same seed). Verdict: the Fisher up-scaling of committed
  coordinates hurts — per-coordinate variance normalization is
  SNR-neutral but makes coordinate noise heavy-tailed (rare informative
  samples dominate), and shaping can't repair per-coordinate tails.
  Consistent with the Adam probe (square-root Fisher ≈ no gain): Fisher-
  style step scaling is not the bottleneck for this search distribution
  (JSONs `..._nat_20261112_120254.json` / `..._nat_20261112_120422.json`). Streaming `SoftES.sample()` also binarises
  row-chunks of 128 now (bool population built incrementally; RNG stream
  changes for populations > 128, statistically equivalent) — pop-1000
  at h1024×1 runs OK with chunk 512 / grad-chunk 128 / pop-chunk 128.
  Pop-1000 probe (h1024×1, standardize, σ=0, he, lr 0.1 = pop-100's lr 1
  scaled 1/P, 300 it, seed 42): val 0.9675 / test 0.9576, still climbing —
  BEATS the best-on-record h1024×2 pop-100 1000-it result (0.9625/0.9530)
  at ~1/3 the iterations and equal-ish wall-clock; SNR ∝ √P confirmed
  empirically
  (`results/es_loss_ce_h1024x1_he_relu_p1000_lr0.1_seed42_t0_shstandardize_20261112_105039.json`).
  Pop-1000 at 1000 it (same recipe): val 0.9762 / test 0.9706 — BEST ES
  RESULT ON RECORD; +1.4pp val / +1.8pp test over h1024×2 pop-100 1000-it,
  +1.9pp/+2.2pp over h1024×1 pop-100 1000-it; gently decelerating at 1000
  (fitness CE −0.076); gap to the Adam-STE ceiling (0.9822/0.9812) narrows
  to ~0.6pp val / ~1.1pp test
  (`results/es_loss_ce_h1024x1_he_relu_p1000_lr0.1_seed42_t0_shstandardize_20261112_111604.json`).
  h256×3 pop-1000 (same recipe, 269k bits, chunk 512 / grad-chunk 32 /
  pop-chunk 128; added 2026-11-12): 300 it → val 0.9642 / test 0.9566;
  1000 it → val 0.9775 / test 0.9697, still climbing (fitness CE −0.056)
  — ties/edges the h1024×1 flagship (0.9762/0.9706) at 1/3 the bits and
  ~3x less compute per iteration; depth compensates for width under the
  big-population recipe
  (`results/es_loss_ce_h256x3_he_relu_p1000_lr0.1_seed42_t0_shstandardize_20261112_132614.json`).
  h1024×3 pop-1000 OOMs at the population bool concat (2.9 GB × 2 during
  tf.concat) — would need a preallocated bool buffer with slice-assign
  (or CPU-resident population) to run.
  Adam probe on h1024×1 (σ=0, standardize, lr sweep 1e-4→10, pop 100, 300 it,
  2026-11-12): 1e-4→0.07, 1e-3→0.09, 1e-2→0.21, 0.1→0.7645, **1→0.9270**,
  3→0.8775, 10→0.0956 (saturation freeze) — Adam's sweet spot is also lr 1
  (its ~lr-bounded steps make the effective ladder ~1 raw unit wide).
  At 300 it Adam is BEHIND raw (0.7417 vs 0.8478 val @ 30; ties 0.9403 vs
  0.9420 @ 300); at 1000 it Adam edges ahead only marginally (val 0.9587 vs
  0.9577, test 0.9512 vs 0.9485, crossover ~it 500; JSONs
  `..._lr1.0_seed42_t0_shstandardize_adam_20261112_103541.json` vs
  `..._shstandardize_20261111_215315.json`). Verdict: Adam ≈ raw here — the
  gain is seed-noise level. With standardize shaping already normalizing the
  advantage globally, per-coordinate step normalization adds nothing
  measurable; the per-generation fitness signal, not step scaling, is the
  bottleneck (momentum's √k SNR argument pays only ~+0.3pp, within seed
  variance). Perturbation table pre-generated (circular; `table_rows=0` passed to
  run_es means the legacy streaming sampler — inside SoftES `None` disables it, a
  literal `0` table crashes). Antithetic default off (consumes `ceil(P/2)`
  rows/generation, pairs `u` with `1-u`; bias pairs mirror around the mean).
  Fitness shaping (`shaping`, added 2026-11-11) replaces the raw advantage
  `f_i - mean(f)` with a bounded utility shared by both coordinate gradients:
  `none` (default, unbiased REINFORCE), `standardize` (population z-score),
  `rank` (centered average ranks in [-0.5, 0.5], Salimans et al. 1703.03864;
  average ranks for ties, all-equal fitness -> zero utility). Shaped utilities
  are population-dependent, hence biased as estimators — they bound the step
  and decouple it from the fitness scale (exploding-CE runaway guard).
  Known deviation: with `antithetic=True` the bias pair contributes
  `(f+ - f-)*eps/sigma^2` (expectation `2*grad/sigma` per pair) instead of the
  paper-normalized `(f+ - f-)*eps/(2*sigma)` — vs OpenAI's per-pair
  normalization the bias steps are inflated by `2/sigma`, but under the repo's
   own population-sum convention the total is `P*grad` either way (SNR-neutral;
   the real antithetic gain is the variance reduction, not the normalization).
- `bes/algs/ges.py`: GaussianES — continuous-latent search distribution for the
  gaussian-paper (added 2026-11-18): `theta in R^D` init from a zero-mean
  Gaussian `N(0, init_std^2)` (exact sign(0) ties have measure zero; bits are
  quantized as `x > 0`), each generation perturbs with isotropic Gaussian noise
  at a SHARED FIXED scale `sigma` (locked decision; no per-coordinate sigma),
  quantizes `sign(theta + sigma*eps)` and evaluates the same `BNNFitness`
  black box as the Bernoulli ES (weights-only, `bias=None`). REINFORCE score
  `(x - theta)/sigma^2 = eps/sigma` reduces to the sampling noise; update
  `theta += lr * sum_i u_i * eps_i / sigma` (population-summed convention,
  score has unit per-coordinate variance vs SoftES's bounded `x - s`, so lr
  units are comparable but tails heavier). The induced bit distribution is
  Bernoulli with `p_j = Phi(theta_j / sigma)` — same family as SoftES under a
  monotone logit reparameterisation (good paper remark). Noise memory: the
  `(P, D)` float perturbations are needed again at update time, so they are
  NEVER materialized whole — `sample()`/`update()` replay the identical
  row-chunked stream, re-created per generation from a splitmix64-mixed seed
  (`_noise_seed`; TF `Generator.state` assignment is silently ignored and
  `from_state` does not reproduce the stream, and raw consecutive seeds
  OVERLAP as philox counters — seed 1 = seed 0 shifted). Table mode
  (`table_rows>0`) pre-generates theta-independent standard-normal rows
  (reusable at every theta like SoftES; update re-gathers the stored
  generation indices). Antithetic available, default off: pairs `eps`/`-eps`
  (bits are NOT complements unlike Bernoulli p=0.5; ceil(P/2) rows/gen, odd
  P adds one unpaired tail). Shaping/adam/grad-chunk mirror SoftES (grad-chunk
  sub-chunks the multiply-reduce only — the noise stream is unchanged, so a
  grad-chunk run is bitwise-identical to the same run without it except fp
  summation order). Diagnostics: entropy/committed over the induced `p`,
  plus Gaussian-specific `flip_frac = E[Phi(-|theta|/sigma)]` (perturbation
  flip probability; the sigma-relative commitment measure); packed layout is
  10 base scalars + 3 per layer (vs SoftES's 9 + 2). Smoke result (h64x1 relu
  he, p100, lr 1, sigma 1, 50 it): val 0.75, committed 66% @ it 50 — commits
  much faster than the Bernoulli ES at comparable lr; expect an lr sweep.
  Wide-net memory note: a (T, D) float32 table is ~T*D*4 bytes (h1024 D=813k:
  T=1024 table = 3.3 GB — use `--table-rows 0` streaming, same gotcha as
  SoftES).
- `bes/algs/mga.py`: MicrobialGA — persistent population (uniform Bernoulli 0.5 weight
  bits, zero biases) evolved by Harvey-style microbial tournaments (port of the PyTorch
  reference `/data/mgab`: ring-deme perfect matching via `pair()`, whole-population
  batch fitness via `BNNFitness`, winner/loser crossover/mutation in `tournaments()`;
  `step(fitness_fn)` = pair + evaluate + tournaments). Losers get uniform crossover
  (`crossrate`, per gene incl. biases) then mutation (weight-bit flips at
  `mutation_rate`, Gaussian bias noise `bias_std`); `deme_size >= P` = fully random
  matching; tournament losers are overwritten in place. Default eval candidate =
  current fitness argmax (`best_bits`/`best_bias`). RNG noise (init population,
  crossover mask, flip mask) is row-chunked now (`_bool_mask`, adaptive chunk
  min(128, 1GB/(12·D)) — same rule as SoftES streaming sampling) so pop-1000 at
  h1024 doesn't OOM; RNG stream changes for wide/populations > chunk (statistically
  equivalent).
- `bes/fitness.py`: `BNNFitness` batch schedules — `schedule="random"` (default,
  without-replacement per call) and `schedule="cycle"` (fixed contiguous
  mini-batches in epoch order, partial last batch, wraps around; mgab
  `score_mode='batch'` tournament protocol — used by `run_mga_sweep.py`).
- `experiments/run_mga_sweep.py`: deme × crossrate × seeds grid runner (mgab
  reproduction; per-seed JSONs under per-cell dirs in the `--out-dir`, default
  `results/mga_sweep_exact/`, plus `summary_cr*.json` with config-means;
  `--resume` skips files whose cached config matches; stale caches fail
  unless `--rerun-stale` is passed). Dense
  monitoring = all-P full-val metrics every `--eval-every`; mgab recorded every
  iteration on its 10k test set (GPU-era patchwork of iteration counts/params —
  the recorded data is not internally uniform; this sweep is a single-protocol
  re-run). `run_mga.py` modernized 2026-11-12: `--hidden/--layers/--activation/
  --chunk/--pop-chunk` flags + arch-tagged filenames. MGA h1024×1 pop-1000
  probe (mgab tournament recipe: deme 256, cr 0.5, mr 1e-4, bstd 0 = biases
  frozen to match ES σ=0, relu + he scale, CE, 300 it, seed 42): val 0.2913 /
  test 0.2800 — slow crawl, no collapse (pop fitness mean −2.91 → −2.46
  monotone), best net stuck 0.21-0.29; MGA is far behind the ES at equal
  budget on this arch (ES pop-1000 300-it: 0.9576 test)
  (`results/mga_loss_ce_h1024_he_relu_p1000_d256_cr0.5_mr0.0001_bstd0.0_seed42_20261112_135334.json`).
- Result archives: `results/mga_sweep/` (cr=1.0 only, scale=none, sign, mr=1e-5,
  bias_std=1e-5, 1000 it — no collapse, plateau ~0.40 all demes),
  `results/mga_sweep_fixed_scale/` (complete demes 2/4/8 + 2/3 of deme 16),
  `results/mga_sweep_exact/` (mgab's exact recorded recipe: relu, no-scale,
  mr=1e-4, bias_std=1, 2000 it; full deme × crossrate grid, 3 seeds — the
  deme-256 cr=1.0 cell REPRODUCES the mgab collapse: peak 0.405 / final 0.180 vs
  mgab 0.377 / 0.169). Figures in `figures/mga_sweep_noscale/` and
  `figures/mga_sweep_exact/`.
  `results/mga_sweep_exact_sign/` (same recipe with `--activation sign`,
  swap from the relu run) — NO collapse: peak ~0.44 / final ~0.44, plateau-like
  behavior (best_net final ≈ peak). Figures in `figures/mga_sweep_exact_sign/`.
  `results/mga_sweep_exact_h1024/` (same recipe with `--hidden 1024`, sign) —
  NO collapse, much better: peak/final ≈ 0.65, test ≈ 0.61-0.63. Results in
  `results/mga_sweep_exact_h1024/` (figures not archived).
- `experiments/plot_mga_sweep.py`: sweep figures — per-crossrate deme trace
  plots + peak/collapse heatmap (deme × crossrate), mirroring mgab's
  `besttest_train_demesize_cr*.png` and `peak_collapse_heatmap.png`.
- Maths: `/data/reinforce_maths.tex` (REINFORCE), `/data/comb-exp/essays/es_derivation.md` (ES derivation).

## Gotchas

- `bes/mlp.py` `weight_shapes` chains hidden widths (`prev -> units`); until
  2026-11-11 every hidden layer got `input_dim` as in-width and multi-hidden
  nets crashed at the first hidden einsum (`run_mga_sweep.py --hidden` was
  single-layer so it was never hit). `tests/test_forward.py` covers
  3-hidden-layer parity (fixed/he/none scale, sign + relu).
- larq import only works with legacy Keras: plain `import larq` crashes (larq
  0.14 vs Keras 3: `tf.keras.layers` has no `LocallyConnected1D`), but
  `TF_USE_LEGACY_KERAS=1` revives it (tf_keras 2.21.0 is already in the venv;
  `tf.keras` then resolves to `tensorflow.keras`). So `experiments/run_ste.py`
  must be run as
  `cd /data/bes && TF_USE_LEGACY_KERAS=1 /data/ste/venv/bin/python experiments/run_ste.py …`.
  Everything else (run_es.py, run_mga.py, run_mga_sweep.py, plot scripts, the
  ES/MGA tests, tests/test_forward.py) never imports larq and works fine
  either way.
- Reproducibility: run scripts call `tf.keras.utils.set_random_seed(args.seed)`;
  the SoftES/MicrobialGA own their TF generators, seeded from `seed`.
- Fitness batch: `BNNFitness` draws a random batch without replacement per call
  (numpy rng, default `batch_size` 4096, clamped to dataset size). Evaluating the
  mode net goes through `mlp.population_metrics` on the full val/test set (as
  `run_es.py` does).

## Conventions

- All algorithms maximise fitness; design decisions are locked in `DESIGN.md` and
  code is authoritative over `DESIGN.md` prose.
- `__init__.py` files stay empty; imports use direct module paths
  (`from bes.algs.es import SoftES`).

## References

- Design: `DESIGN.md`; literature: `papers/bnn_optimization_literature.md`;
  STE source of truth: `/data/ste/train_mnist_bnn.py`; soft-ES reference:
  `/data/comb-exp/algs/es.py`; microbial-GA reference (PyTorch):
  `/data/mgab` (`PopMLP.py`, `geography.py`).
- Credential/push policy: see the parent-level `/data/AGENTS.md` (never push or
  force-push without explicit user confirmation).
