"""Collapse figure for the crossrate-0.5 arc-1-recipe sweep.

Layout (2x2, bottom row spans both columns):
  top-left:  mean best_val_acc vs iteration, one trace per deme
  top-right: mean best_val_loss vs iteration (log y), same deme encoding
  bottom:    peak->final dumbbell of best_val_acc per deme (mean bar + seed dots)

Deme-size encoding: viridis over log2(deme) with a shared colourbar.

Usage:
  /path/to/python experiments/plot_cr05_collapse.py [--in-dir results/mga_sweep_exact]
      [--cr 0.5] [--out figures/cr05_collapse.pdf]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cells(in_dir, cr):
    cells = {}
    for cell_dir in sorted(Path(in_dir).glob(f"p*_d*_b*_cr{cr}")):
        deme = int(cell_dir.name.split("_d")[1].split("_")[0])
        acc, loss = [], []
        steps = None
        for seed_file in sorted(cell_dir.glob(f"{cell_dir.name}_s*.json")):
            j = json.loads(seed_file.read_text())
            if steps is None:
                steps = [h["step"] for h in j["history"]]
            acc.append([h["best_val_acc"] for h in j["history"]])
            loss.append([h["best_val_loss"] for h in j["history"]])
        if acc:
            cells[deme] = (np.array(acc), np.array(loss), np.array(steps))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=str, default=str(ROOT / "results/mga_sweep_exact"))
    parser.add_argument("--cr", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=str(ROOT / "figures/cr05_collapse.pdf"))
    args = parser.parse_args()

    cells = load_cells(args.in_dir, args.cr)
    demes = sorted(cells)
    n_deme = len(demes)
    if n_deme == 0:
        raise SystemExit(f"no cr={args.cr} cells found under {args.in_dir}")
    alleles = np.log2(demes)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(alleles.min()), vmax=float(alleles.max()))
    colors = {d: cmap(norm(float(np.log2(d)))) for d in demes}

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.28)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_db = fig.add_subplot(gs[1, :])

    for d in demes:
        acc, loss, steps = cells[d]
        mean_acc, mean_loss = acc.mean(axis=0), loss.mean(axis=0)
        ax_acc.plot(steps, mean_acc, color=colors[d], lw=2.0, label=str(d))
        ax_loss.plot(steps, mean_loss, color=colors[d], lw=2.0, label=str(d))

    ax_acc.set_title("Mean Validation Accuracy of Best Individual")
    ax_acc.set_ylabel("Validation Accuracy")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Mean Validation Loss of Best Individual")
    ax_loss.set_ylabel("Validation Loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("iteration")
    ax_acc.axhline(0.1, color="gray", lw=0.8, ls=":", alpha=0.8)
    ax_loss.legend(frameon=False, loc="upper right", fontsize=7,
                   title="deme", title_fontsize=7, ncol=1)

    # bars: mean peak validation accuracy per deme; whiskers = seed min-max
    peaks_mean = [cells[d][0].max(axis=1).mean() for d in demes]
    peaks_spread = [cells[d][0].max(axis=1) for d in demes]

    top = max(max(peaks_mean), max(p.max() for p in peaks_spread)) * 1.08
    ax_db.set_xlim(-0.6, n_deme - 0.4)
    ax_db.set_ylim(0.0, max(0.3, top))
    ax_db.set_xticks(range(n_deme))
    ax_db.set_xticklabels([str(d) for d in demes])
    ax_db.axhline(0.1, color="gray", lw=0.8, ls=":", alpha=0.8)
    for i, d in enumerate(demes):
        ax_db.bar(i, peaks_mean[i], width=0.72, color="#4C72B0", zorder=2)
        ax_db.errorbar(i, peaks_mean[i],
                       yerr=[[peaks_mean[i] - peaks_spread[i].min()],
                             [peaks_spread[i].max() - peaks_mean[i]]],
                       color="black", fmt="none", capsize=3, lw=1, zorder=4)
    ax_db.set_title("Mean Peak Validation Accuracy of Best Individual Across Deme Sizes")
    ax_db.set_xlabel("Deme Size")
    ax_db.set_ylabel("Validation Accuracy")
    ax_db.grid(alpha=0.25, axis="y", lw=0.5)

    fig.suptitle("Deme Size Controls Peak Fitness in Deceptive Landscape", fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
