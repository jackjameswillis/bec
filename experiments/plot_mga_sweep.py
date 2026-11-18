"""Figures for the deme-size x crossrate microbial-GA sweep.

Reads the per-seed JSONs written by ``experiments/run_mga_sweep.py`` from
``results/mga_sweep_exact/`` (or ``--in-dir``) and writes into ``figures/mga_sweep/``:

- one mean-over-seeds trace figure per crossrate (best val accuracy vs step,
  one line per deme), mirroring the mgab ``besttest_train_demesize_cr*`` plots;
- the (deme, crossrate) heatmap of peak mean-over-seeds best val accuracy and
  the collapse magnitude (peak - final), mirroring mgab's summary heatmap.

Usage:
  /path/to/python experiments/plot_mga_sweep.py [--in-dir results/mga_sweep_exact] [--out-dir figures/mga_sweep]
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

COLOURS = plt.get_cmap("tab10")


def load_traces(in_dir):
    cells = defaultdict(dict)
    for path in sorted(Path(in_dir).rglob("p*_d*_b*_cr*_s*.json")):
        rec = json.loads(path.read_text())
        cfg = rec["config"]
        cells[(cfg["crossrate"], cfg["deme"])][cfg["seed"]] = rec
    return cells


def mean_trace(seeded, key):
    n_steps = min(len(rec["history"]) for rec in seeded.values())
    series = np.stack([np.array([h[key] for h in rec["history"][:n_steps]])
                       for rec in seeded.values()])
    steps = [h["step"] for h in seeded[sorted(seeded)[0]]["history"][:n_steps]]
    return steps, series.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    in_dir = Path(args.in_dir) if args.in_dir else root / "results" / "mga_sweep_exact"
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures" / "mga_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_traces(in_dir)
    if not cells:
        raise SystemExit(f"no sweep JSONs found under {in_dir}")
    crossrates = sorted({cr for cr, _ in cells})
    demes = sorted({d for _, d in cells})
    print(f"{len(cells)} cells, demes={demes}, crossrates={crossrates}")

    # 1. per-crossrate trace figures
    for cr in crossrates:
        cr_demes = sorted(d for c, d in cells if c == cr)
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, deme in enumerate(cr_demes):
            steps, series = mean_trace(cells[(cr, deme)], "best_val_acc")
            ax.plot(steps, series, lw=2, color=COLOURS(k), label=f"deme={deme}")
        ax.set_title(f"Mean best val accuracy (crossrate={cr})")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best val accuracy")
        ax.grid(alpha=0.3)
        ax.legend(title="demesize")
        fig.tight_layout()
        png = out_dir / f"val_demesize_cr{cr}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"wrote {png}")

    # 2. peak / collapse heatmap over (deme, crossrate)
    def grid(key_peak, key_final=None):
        rows = np.full((len(demes), len(crossrates)), np.nan)
        for i, deme in enumerate(demes):
            for j, cr in enumerate(crossrates):
                seeded = cells.get((cr, deme))
                if not seeded:
                    continue
                if key_final is None:
                    fp = [key_peak(rec) for rec in seeded.values()]
                    rows[i, j] = np.mean(fp)
                else:
                    p = [key_peak(rec) for rec in seeded.values()]
                    f = [key_final(rec) for rec in seeded.values()]
                    rows[i, j] = np.mean(p) - np.mean(f)
        return rows

    peak = grid(lambda rec: max(h["best_val_acc"] for h in rec["history"]))
    final = grid(lambda rec: rec["history"][-1]["best_val_acc"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in (
        (axes[0], peak, "Peak mean best val accuracy", "{:.2f}"),
        (axes[1], peak - final, "Collapse magnitude (peak - final)", "{:.2f}"),
    ):
        im = ax.imshow(data, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("crossrate")
        ax.set_ylabel("demesize")
        ax.set_xticks(range(len(crossrates)), [str(c) for c in crossrates])
        ax.set_yticks(range(len(demes)), [str(d) for d in demes])
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
        for i in range(len(demes)):
            for j in range(len(crossrates)):
                val = data[i, j]
                if not np.isfinite(val):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            color="w", fontsize=8)
                    continue
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="w" if val < vmax * 0.7 else "k", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png = out_dir / "heatmap.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}")

    # 3. summary table
    print("\n=== SUMMARY: mean-over-seeds best val accuracy (cr | deme) ===")
    print("cr      " + "".join(f"{d:>8}" for d in demes))
    for cr in crossrates:
        row = f"{cr:>6} "
        for deme in demes:
            seeded = cells.get((cr, deme))
            if not seeded:
                row += f"{'n/a':>8}"
                continue
            _, series = mean_trace(seeded, "best_val_acc")
            row += f"{max(series):8.4f}"
        print(row + "   [peak]")
    for cr in crossrates:
        row = f"{cr:>6} "
        for deme in demes:
            seeded = cells.get((cr, deme))
            if not seeded:
                row += f"{'n/a':>8}"
                continue
            _, series = mean_trace(seeded, "best_val_acc")
            row += f"{series[-1]:8.4f}"
        print(row + "   [final]")


if __name__ == "__main__":
    main()
