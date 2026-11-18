"""Figures for a population/convergence plan (rank + shaping stages).

Reads the diagnostic run JSONs written by ``experiments/run_es.py
--diagnostics`` and writes PDF+PNG pairs into the output directory (default
``figures/convergence/``):

- fig_pop_scaling  final val/test accuracy vs population (log-x, 5-seed
  error bars) and per-bit entropy traces vs iteration, one curve per
  population (mean +/- std over seeds);
- fig_entropy      per-bit entropy (log-y) and committed fraction vs
  iteration, one curve per population (mean +/- std over seeds);
- fig_p_hist       p-histogram snapshot grid (rows = population, columns =
  iterations 1 / 50 / 500 / 2000);
- fig_shaping_clock  (only when >= 2 runs per shaping arm at the comparison
  cell) entropy curves for standardize vs rank plus final accuracy bars.

Usage:
  /path/to/python experiments/plot_convergence.py \
      [--in-dir results] [--out-dir figures/convergence] [--tag h256x2]
      [--pattern GLOB] [--iterations 2000] [--hist-cols 1,50,500,2000]
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
COLOURS = plt.get_cmap("tab10")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default=None,
                   help="run directory (default: <repo>/results)")
    p.add_argument("--out-dir", default=None,
                   help="figure directory (default: <repo>/figures/convergence)")
    p.add_argument("--tag", default="h256x2",
                   help="filename tag of the plan runs (es_<tag>_...)")
    p.add_argument("--pattern", default=None,
                   help="glob for run JSONs (default: "
                        "*<tag>*_p*_lr*_seed*_t0_diag<iterations>.json)")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--hist-cols", default="1,50,500,2000",
                   help="snapshot iterations shown in fig_p_hist")
    return p.parse_args()


def load_arms(in_dir, pattern, iterations):
    """-> {shaping: {population: [record, ...]}} for diagnostic runs."""
    arms = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(in_dir).glob(pattern)):
        rec = json.loads(path.read_text())
        cfg = rec["config"]
        if len(rec.get("diagnostics", [])) < iterations:
            continue
        arms[cfg["shaping"]][cfg["population"]].append(rec)
    return arms


def seed_std(x, axis=None):
    a = np.asarray(x)
    if a.shape[0] < 2:
        return np.zeros_like(np.mean(a, axis=axis))
    return np.std(a, axis=axis, ddof=1)


def acc_panel(ax, runs_by_pop):
    pops = sorted(runs_by_pop)
    x = np.arange(len(pops))
    for i, key in enumerate(("val", "test")):
        mean = [np.mean([r[key]["accuracy"] for r in runs_by_pop[p]]) for p in pops]
        std = [seed_std([r[key]["accuracy"] for r in runs_by_pop[p]]) for p in pops]
        ax.errorbar(x, 100 * np.array(mean), yerr=100 * np.array(std),
                    marker="o" if key == "test" else "s", capsize=3,
                    color=f"C{i}", label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pops])
    ax.set_xlabel("population size")
    ax.set_ylabel("accuracy (%)")
    ax.legend()


def entropy_panel(ax, runs_by_pop, iterations):
    """Per-bit entropy vs iteration, one mean+/-std curve per population."""
    for j, p in enumerate(sorted(runs_by_pop)):
        runs = runs_by_pop[p]
        series = np.stack([
            np.array([d["entropy_bits"] for d in r["diagnostics"][:iterations]])
            for r in runs])
        mean, std = series.mean(axis=0), seed_std(series, axis=0)
        its = np.arange(1, iterations + 1)
        ax.plot(its, mean, color=f"C{j}", label=f"$P={p}$")
        ax.fill_between(its, mean - std, mean + std, color=f"C{j}",
                        alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-bit entropy (bits)")
    ax.legend()


def entropy_panels(fig, runs_by_pop, iterations):
    for i, (key, ylabel, ylog) in enumerate(
            (("entropy_bits", "per-bit entropy (bits)", True),
             ("committed_frac", "committed fraction", False))):
        ax = fig.add_subplot(1, 2, i + 1)
        for j, p in enumerate(sorted(runs_by_pop)):
            runs = runs_by_pop[p]
            series = np.stack([
                np.array([d[key] for d in r["diagnostics"][:iterations]])
                for r in runs])
            mean, std = series.mean(axis=0), seed_std(series, axis=0)
            its = np.arange(1, iterations + 1)
            ax.plot(its, mean, color=f"C{j}", label=f"$P={p}$")
            ax.fill_between(its, mean - std, mean + std, color=f"C{j}",
                            alpha=0.2, lw=0)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        if ylog:
            ax.set_yscale("log")
        ax.legend()


def hist_grid(fig, runs_by_pop, cols):
    pops = sorted(runs_by_pop)
    for i, p in enumerate(pops):
        for j, step in enumerate(cols):
            ax = fig.add_subplot(len(pops), len(cols), i * len(cols) + j + 1)
            snap = next((
                s for rec in runs_by_pop[p] for s in rec.get("p_snapshots", [])
                if s["step"] == step), None)
            if snap is None:
                ax.axis("off")
                continue
            counts = np.array(snap["hist"], dtype=float)
            counts /= max(counts.sum(), 1)
            ax.bar(np.arange(len(counts)), counts, width=1.0, color=f"C{i}")
            ax.set_xlim(-0.5, len(counts) - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"it {step}")
            if j == 0:
                ax.set_ylabel(f"$P={p}$")


def shaping_panel(fig, arms, cell_pop):
    """Entropy traces rank vs standardize at the comparison cell + final bars.

    Returns False (and leaves ``fig`` empty) when either arm has < 2 runs.
    """
    shapes = [s for s in ("rank", "standardize") if s in arms
              and len(arms[s].get(cell_pop, [])) >= 2]
    if len(shapes) < 2:
        print(f"fig_shaping_clock skipped: need >=2 runs per shaping arm at "
              f"P={cell_pop}; have "
              + str({s: len(arms[s].get(cell_pop, [])) for s in arms}))
        return False
    ax = fig.add_subplot(1, 2, 1)
    iters = None
    for j, shape in enumerate(shapes):
        runs = arms[shape][cell_pop]
        series = np.stack([
            np.array([d["entropy_bits"] for d in r["diagnostics"]])
            for r in runs])
        mean, std = series.mean(axis=0), seed_std(series, axis=0)
        iters = np.arange(1, series.shape[1] + 1)
        ax.plot(iters, mean, color=f"C{j}", label=shape)
        ax.fill_between(iters, mean - std, mean + std, color=f"C{j}",
                        alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-bit entropy (bits)")
    ax.set_title(f"entropy, $P={cell_pop}$")
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    for j, shape in enumerate(shapes):
        runs = arms[shape][cell_pop]
        t = 100 * np.array([r["test"]["accuracy"] for r in runs])
        ax.bar(j, t.mean(), yerr=t.std(), capsize=3, color=f"C{j}")
        ax.text(j, t.mean() + t.std(), f"{t.mean():.2f}", ha="center",
                va="bottom")
    lo = min(100 * np.mean([r["test"]["accuracy"] for r in arms[s][cell_pop]])
             for s in shapes) - 0.6
    hi = max(100 * np.mean([r["test"]["accuracy"] for r in arms[s][cell_pop]])
             for s in shapes) + 0.6
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes)
    ax.set_ylabel("final test accuracy (%)\n[axis truncated]")
    ax.set_title(f"final test accuracy, $P={cell_pop}$")
    return True


def save(fig, out_dir, name):
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / (name + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / (name + ".pdf"))


def main():
    args = parse_args()
    in_dir = Path(args.in_dir) if args.in_dir else ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "figures" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = args.pattern or (
        f"*{args.tag}*_p*_lr*_seed*_t0_diag{args.iterations}.json")
    arms = load_arms(in_dir, pattern, args.iterations)
    if not arms:
        raise SystemExit(f"no diagnostic run JSONs matching {pattern!r} "
                         f"under {in_dir}")
    for shape in sorted(arms):
        print(f"arm '{shape}': populations "
              + str({p: len(r) for p, r in sorted(arms[shape].items())}))
    all_runs = [r for by_pop in arms.values()
                for runs in by_pop.values() for r in runs]
    for field in ("hidden", "layers", "scale", "activation"):
        values = {r.get("config", {}).get(field) for r in all_runs}
        if len(values) > 1:
            print(f"warning: matched runs have mixed {field}: "
                  f"{sorted(values, key=str)}")

    rank = arms.get("rank") or next(iter(arms.values()))

    fig = plt.figure(figsize=(7.0, 2.8))
    acc_panel(fig.add_subplot(1, 2, 1), rank)
    entropy_panel(fig.add_subplot(1, 2, 2), rank, args.iterations)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle("larger populations perform better and converge slower")
    save(fig, out_dir, "fig_pop_scaling")

    fig = plt.figure(figsize=(7.0, 2.8))
    entropy_panels(fig, rank, args.iterations)
    fig.suptitle("search-distribution convergence")
    save(fig, out_dir, "fig_entropy")

    fig = plt.figure(figsize=(7.0, 1.6 * len(rank)))
    hist_grid(fig, rank, [int(c) for c in args.hist_cols.split(",")])
    save(fig, out_dir, "fig_p_hist")

    if "standardize" in arms:
        cell = max(max(arms["rank"]), max(arms["standardize"]))
        fig = plt.figure(figsize=(7.0, 2.8))
        if shaping_panel(fig, arms, cell):
            save(fig, out_dir, "fig_shaping_clock")


if __name__ == "__main__":
    main()
