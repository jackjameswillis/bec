# Retained results

Only result files used by the Bernoulli paper are kept in the working tree. The
complete pre-prune archive is:

- `/data/bes_results_archive_20261118.tar.zst`
- SHA-256 in `/data/bes_results_archive_20261118.tar.zst.sha256`

## Retained files

- `es_h256x2_he_relu_p{256,512,1024,2048}_lr*_shrank_seed{42..46}_t0_diag2000.json`
  — population-scaling figure (`fig_pop_scaling`).
- `es_h256x2_he_relu_p2048_lr0.048828125_shrank_seed{42..46}_t0_diag5000.json`
  — long training traces, ES numbers in the ES-vs-STE table.
- `es_h256x2_he_relu_p2048_lr0.048828125_shrank_seed42_t0_diag5000_theta.npz`
  — per-layer Bernoulli-expectation histograms.
- `ste_h256x2_he_noiq_adam_seed{42..46}.json`
  — STE numbers in the ES-vs-STE table and the STE comparison figure.

## Regenerating equivalent figures

```bash
/data/ste/venv/bin/python experiments/plot_convergence.py \
    --in-dir results --out-dir figures/convergence
/data/ste/venv/bin/python experiments/plot_long_ste.py \
    --in-dir results --out-dir figures/convergence
/data/ste/venv/bin/python experiments/plot_theta_expectations.py \
    --npz results/es_h256x2_he_relu_p2048_lr0.048828125_shrank_seed42_t0_diag5000_theta.npz \
    --out figures/theta_expectations_layers_h256x2_p2048_seed42
```

The tracked paper copies live under `submission/bernoulli-paper/`; those are the
canonical paper figures. Current plotting code may produce minor styling
differences. Raw runs can be reproduced with
`experiments/run_h256x2_rank_plan.sh`.
