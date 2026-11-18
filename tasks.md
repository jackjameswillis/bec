# Remaining tasks

Cleanup pass completed: shared ES helpers (`bes/algs/_common.py`), shared runner
plumbing (`experiments/_common.py`), shared `run_search` loop for `run_es` /
`run_ges`, validation and `table_rows=0` handling, sparse-label CE, STE cleanup,
plot-script fixes, dead-code removal, and documentation corrections.

## High priority

- [x] Create the initial git commit and decide repository tracking. Source,
      tests, docs, and LaTeX sources are tracked; `results/` and `figures/`
      remain ignored.
- [x] Validate the shared `run_search` refactor against a known archived run:
      run a short seed-matched recipe (h256x2, p2048, rank, 200 it) and compare
      history, diagnostics, snapshots, and step-200 metrics with the archived
      JSON; all match exactly.
- [x] Add an aggregate test runner (`tests/run_all.py`) and optional smoke tests
      for the runner/plotting entry points; no CI, but local `tests/run_all.py`
      now exists.
- [ ] Prune or archive `results/`: 67 MB, ~296 JSON files, duplicate logs and
      old timestamped variants. Decide whether any summaries/figures should be
      tracked despite the current `.gitignore`.
- [ ] Audit `AGENTS.md` again after the code refactor. It is ~37 KB and mixes
      commands, architecture, and experiment history; consider splitting or
      trimming it now that shared modules and defaults changed.

## Medium priority

- [x] Add resume validation to `run_paper_sweep.py` and `run_mga_sweep.py`:
      compare cached JSON configs with the requested recipe before skipping, so
      stale configs are not silently reused.
- [x] Reduce rank-shaping memory: `fitness_utility` now uses an argsort-based
      average-rank implementation with O(P) memory, bit-identical to the old
      `(P, P)` reference.
- Dropped: guard perturbation-table memory; an OOM crash is an acceptable
      failure mode.
- [x] Make STE `--use-bias` match the ES forward exactly for
      `--scale != none`: biases are now added after the per-layer scale.
- [x] Improve `run_ste.py` output provenance: default filenames now include
      learning rate, epochs, and batch size.
- [x] Rename `bes/mlp.net_logits` (it returns softmax probabilities) to
      `net_probs`, keeping `net_logits` as a backward-compatible alias.
- [x] Clean old theta `.npz` archives that still contain unused `mode_it_*`
      arrays from before the theta-dump cleanup.

## Low priority

- [x] `experiments/run_h256x2_rank_plan.sh` now accepts `PY=/path/to/python`
      and falls back to `/data/ste/venv/bin/python`.
- [x] `plot_convergence.py` and `plot_long_ste.py` accept pattern/glob overrides;
      long-run labels and titles derive from the loaded configs.
- [ ] Deduplicate `submission/bernoulli-paper` and `submission/gaussian-paper`
      LaTeX assets (`llncs.cls`, `splncs04.bst`, `refs.bib`).
- [ ] Fill in or remove the remaining `TODO` placeholders in
      `submission/gaussian-paper/total.tex` and remove any stale generated PDFs.
- Dropped: generate the missing `mga_sweep_exact_h1024` figures; update or
      remove the documentation reference during the `AGENTS.md` audit.
- [x] Add a compatibility test for the changed default output filenames
      (now include `_s{sigma}` and `_it{iterations}` in `run_es` / `run_ges`).
