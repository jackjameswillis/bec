"""Long-run figures and the ES-vs-STE comparison.

Reads run JSONs from ``results/`` (default patterns match the h256x2/P2048
paper plan):

- fig_long_runs   mode-net validation trace over ``--iterations`` with the
  tuned Adam-STE mean as a ceiling guide, plus the search-distribution
  entropy and committed-fraction tails over the same horizon;
- fig_ste_comparison  final test accuracy (mean +/- std) for the ES long
  runs vs the tuned Adam-STE recipe, wall-clock annotated under each bar.

Usage:
  /path/to/python experiments/plot_long_ste.py [--in-dir results]
      [--out-dir figures/convergence] [--es-glob GLOB] [--ste-glob GLOB]
      [--iterations 5000]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "legend.fontsize": 8, "axes.grid": True,
                     "grid.alpha": 0.3})

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default=None,
                   help="run directory (default: <repo>/results)")
    p.add_argument("--out-dir", default=None,
                   help="figure directory (default: <repo>/figures/convergence)")
    p.add_argument("--es-glob", default=None,
                   help="glob for long ES runs (default: "
                        "es_h256x2_*_p2048_lr0.048828125_shrank_seed*"
                        "_t0_diag<iterations>.json)")
    p.add_argument("--ste-glob", default=None,
                   help="glob for STE comparison runs "
                        "(default: ste_h256x2_he_noiq_*_seed*.json)")
    p.add_argument("--iterations", type=int, default=5000,
                   help="long-run horizon and minimum diagnostics length")
    return p.parse_args()


def load_shrank(in_dir, iterations, pattern=None):
    pattern = pattern or (
        "es_h256x2_*_p2048_lr0.048828125_shrank_seed*"
        f"_t0_diag{iterations}.json")
    recs = []
    for path in sorted(Path(in_dir).glob(pattern)):
        rec = json.loads(path.read_text())
        if len(rec.get("diagnostics", [])) >= iterations:
            recs.append(rec)
    return recs


def load_ste(in_dir, pattern=None):
    ste = defaultdict(list)
    pattern = pattern or "ste_h256x2_he_noiq_*_seed*.json"
    for path in sorted(Path(in_dir).glob(pattern)):
        rec = json.loads(path.read_text())
        cfg = rec["config"]
        key = "adam" if cfg["optimizer"] == "adam" else f"sgd lr {cfg['learning_rate']:g}"
        ste[key].append(rec)
    return ste


def warn_mixed(recs, fields):
    if len(recs) < 2:
        return
    for field in fields:
        values = {r.get("config", {}).get(field) for r in recs}
        if len(values) > 1:
            print(f"warning: long runs have mixed {field}: "
                  f"{sorted(values, key=str)}")


def save(fig, out_dir, name):
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / (name + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / (name + ".pdf"))


def fig_long_runs(long_recs, ste, out_dir, iterations):
    cfg = long_recs[0].get("config", {})
    pop = cfg.get("population")
    shaping = cfg.get("shaping")
    label = "ES mode net"
    if pop is not None:
        label = f"ES mode net ($P{{=}}{pop}$"
        if shaping:
            label += f", {shaping}"
        label += ")"

    fig = plt.figure(figsize=(7.0, 2.8))
    ax = fig.add_subplot(1, 2, 1)
    series = np.stack([np.array([h["mode_val_acc"] for h in r["history"]])
                       for r in long_recs])
    steps = [h["step"] for h in long_recs[0]["history"]]
    mean, std = series.mean(axis=0), series.std(axis=0)
    ax.plot(steps, 100 * mean, color="C0", label=label)
    ax.fill_between(steps, 100 * (mean - std), 100 * (mean + std),
                    color="C0", alpha=0.2, lw=0)
    adam = ste.get("adam", [])
    if adam:
        adam_val = 100 * np.mean([r["evals"][-1]["val_acc"] for r in adam])
        ax.axhline(adam_val, ls="--", color="C1", lw=1,
                   label=f"Adam-STE mean ({adam_val:.2f})")
    final_t = 100 * np.mean([r["test"]["accuracy"] for r in long_recs])
    ax.annotate(f"final test {final_t:.2f}", xy=(steps[-1], 100 * mean[-1]),
                xytext=(-95, -14), textcoords="offset points", color="C0")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mode-net val accuracy (%)")
    ax.set_title(f"long runs ({iterations} it)")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(1, 2, 2)
    ent = np.stack([np.array([d["entropy_bits"] for d in r["diagnostics"]])
                    for r in long_recs])
    com = np.stack([np.array([d["committed_frac"] for d in r["diagnostics"]])
                    for r in long_recs])
    its = np.arange(1, ent.shape[1] + 1)
    e_mean, e_std = ent.mean(axis=0), ent.std(axis=0)
    c_mean, c_std = com.mean(axis=0), com.std(axis=0)
    ln1, = ax.plot(its, e_mean, color="C3", label="entropy (left, log)")
    ax.fill_between(its, e_mean - e_std, e_mean + e_std, color="C3",
                    alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-bit entropy (bits)", color="C3")
    ax.tick_params(axis="y", colors="C3")
    ax2 = ax.twinx()
    ln2, = ax2.plot(its, c_mean, color="C2", label="committed (right)")
    ax2.fill_between(its, c_mean - c_std, c_mean + c_std, color="C2",
                     alpha=0.2, lw=0)
    ax2.set_ylabel("committed fraction", color="C2")
    ax2.tick_params(axis="y", colors="C2")
    ax2.grid(False)
    ax.legend(handles=[ln1, ln2], loc="center right")
    ax.set_title("commitment tail over the long horizon")
    save(fig, out_dir, "fig_long_runs")


def fig_ste_comparison(long_recs, ste, out_dir, iterations):
    groups = [(f"ES\n{iterations} it",
               [r["test"]["accuracy"] for r in long_recs], "C0")]
    adam = ste.get("adam", [])
    if adam:
        groups.append(("Adam-STE", [r["test"]["accuracy"] for r in adam], "C1"))
    if not groups:
        return
    walls = [f"{np.mean([r['wall_time_s'] for r in recs]):.0f} s"
             for recs in (long_recs, adam)]
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    means = [100 * np.mean(t) for _, t, _ in groups]
    stds = [100 * np.std(t) for _, t, _ in groups]
    for j, ((lab, t, col), m, s) in enumerate(zip(groups, means, stds)):
        ax.bar(j, m, yerr=s, capsize=3, color=col)
        ax.text(j, m, f"{m:.2f}", ha="center", va="bottom")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[0]}\n[{w}]" for g, w in zip(groups, walls)])
    ax.set_ylim(min(m - s for m, s in zip(means, stds)) - 1.0,
                min(100.0, max(m + s for m, s in zip(means, stds)) + 0.5))
    ax.set_ylabel("final test accuracy (%)")
    cfg = long_recs[0].get("config", {})
    hidden, layers = cfg.get("hidden"), cfg.get("layers")
    title = "evolutionary strategy vs straight-through estimator"
    if hidden is not None and layers is not None:
        title += f" (h{hidden}$\\times${layers}, same architecture)"
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out_dir, "fig_ste_comparison")


def fig_val_traces(long_recs, out_dir):
    fig = plt.figure(figsize=(6.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    series = np.stack([np.array([h["mode_val_acc"] for h in r["history"]])
                       for r in long_recs])
    steps = [h["step"] for h in long_recs[0]["history"]]
    for k in range(series.shape[0]):
        ax.plot(steps, 100 * series[k], color=f"C{k}", alpha=0.3, lw=0.8)
    mean = 100 * series.mean(axis=0)
    ax.plot(steps, mean, color="C0", lw=1.8)
    lo, hi = 100 * series.min(), 100 * series.max()
    pad = max(0.1, 0.05 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("training curves")
    save(fig, out_dir, "fig_val_traces")


def main():
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    in_dir = Path(args.in_dir) if args.in_dir else ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "figures" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_recs = load_shrank(in_dir, args.iterations, args.es_glob)
    ste = load_ste(in_dir, args.ste_glob)
    warn_mixed(long_recs, ("hidden", "layers", "scale", "activation",
                           "population", "shaping"))
    print(f"long {len(long_recs)} | STE groups: "
          + str({k: len(v) for k, v in sorted(ste.items())}))
    if not long_recs:
        raise SystemExit(
            f"no runs with >= {args.iterations} diagnostics found")
    fig_long_runs(long_recs, ste, out_dir, args.iterations)
    fig_ste_comparison(long_recs, ste, out_dir, args.iterations)
    fig_val_traces(long_recs, out_dir)


if __name__ == "__main__":
    main()
