"""Figures for the paper sweep (population / depth / fitness shaping / batch).

Reads the sweep outputs written by ``experiments/run_paper_sweep.py``
(``summary.json`` plus the per-seed run JSONs under ``runs/``) and writes
PDF+PNG pairs into ``figures/paper_sweep/``:

- fig_pop_scaling  final val/test accuracy vs population (log-x, 3-seed
  error bars) and wall-clock vs population with a linear-cost guide;
- fig_depth        val-accuracy traces for h256 x {1,2,3} with the h1024x1
  base cell as a width reference (matched population/lr/batch), plus final
  test-accuracy bars;
- fig_shaping      traces for none/standardize/rank, plus final bars;
- fig_batch        final val/test accuracy vs fitness batch size (log-x);
- fig_overview     2x2 single-figure version of the four panels.

Usage:
  /path/to/python experiments/plot_paper_sweep.py [--in-dir results/paper_sweep] [--out-dir figures/paper_sweep]
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

COLOURS = plt.get_cmap("tab10")


def load_runs(in_dir):
    runs = defaultdict(list)
    for path in sorted((in_dir / "runs").glob("*.json")):
        rec = json.loads(path.read_text())
        cfg = rec.get("config", {})
        try:
            key = (cfg["hidden"], cfg["layers"], cfg["population"],
                   cfg["fitness_batch"], cfg["shaping"])
        except KeyError:
            continue
        runs[key].append(rec)
    if not runs:
        raise SystemExit(f"no run JSONs found under {in_dir / 'runs'}")
    return runs


def traces(seeded, key):
    n = min(len(r["history"]) for r in seeded)
    series = np.stack([np.array([h[key] for h in r["history"][:n]])
                       for r in seeded])
    steps = [h["step"] for h in seeded[0]["history"][:n]]
    std = series.std(axis=0, ddof=1) if len(seeded) > 1 else np.zeros(n)
    return steps, series.mean(axis=0), std


def num_params(hidden, layers):
    dims = [784] + [hidden] * layers + [10]
    return sum(a * b for a, b in zip(dims, dims[1:]))


def run_cell(runs, hidden, layers, pop, batch, shaping):
    return sorted(runs.get((hidden, layers, pop, batch, shaping), []),
                  key=lambda r: r["config"]["seed"])


def bar_panel(ax, section, labels, ticklabels, title, colour):
    means = [section[l]["final_test_acc_mean"] for l in labels]
    stds = [section[l]["final_test_acc_std"] for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3, width=0.6,
           color=colour, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(labels)), ticklabels)
    lo = min(m - s for m, s in zip(means, stds)) - 0.004
    hi = max(m + s for m, s in zip(means, stds)) + 0.002
    ax.set_ylim(lo, hi)
    ax.set_ylabel("final test accuracy")
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.0008, f"{m * 100:.2f}", ha="center", fontsize=7.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--ref-dir", type=str, default=None,
                        help="extra sweep dir for reference traces (e.g. the "
                             "h1024 sweep, overlaid on the depth figure)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    in_dir = Path(args.in_dir) if args.in_dir else root / "results" / "paper_sweep"
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures" / "paper_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((in_dir / "summary.json").read_text())
    runs = load_runs(in_dir)
    ref_runs = load_runs(Path(args.ref_dir)) if args.ref_dir else {}

    pop_section = summary.get("pop", {})
    depth_section = summary.get("depth", {})
    batch_section = summary.get("batch", {})
    shape_section = summary.get("shape", {})

    pop_labels = sorted(pop_section,
                        key=lambda l: pop_section[l]["config"]["pop"])
    depth_labels = sorted(depth_section,
                          key=lambda l: depth_section[l]["config"]["layers"])
    batch_labels = sorted(batch_section,
                          key=lambda l: batch_section[l]["config"]["batch"])
    shape_labels = [l for l in ("sh_none", "sh_standardize", "sh_rank")
                    if l in shape_section]

    def save(fig, name):
        for ext in ("pdf", "png"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)

    def pop_ax(ax, wide=False):
        xs = [pop_section[l]["config"]["pop"] for l in pop_labels]
        for key, marker, label in (("final_val_acc", "s", "val"),
                                   ("final_test_acc", "o", "test")):
            mean = [pop_section[l][key + "_mean"] for l in pop_labels]
            std = [pop_section[l][key + "_std"] for l in pop_labels]
            ax.errorbar(xs, mean, yerr=std, marker=marker, ms=4, capsize=3,
                        lw=1.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(p) for p in xs])
        vals = [v for l in pop_labels
                for k in ("final_val_acc", "final_test_acc")
                for v in (pop_section[l][k + "_mean"] - pop_section[l][k + "_std"],
                          pop_section[l][k + "_mean"] + pop_section[l][k + "_std"])]
        ax.set_ylim(min(vals) - 0.004, max(vals) + 0.002)
        ax.set_xlabel("population P")
        ax.set_ylabel("final accuracy")
        ax.legend()
        if wide:
            ax.set_title("population scaling")

    def batch_ax(ax):
        xs = [batch_section[l]["config"]["batch"] for l in batch_labels]
        for key, marker, label in (("final_val_acc", "s", "val"),
                                   ("final_test_acc", "o", "test")):
            mean = [batch_section[l][key + "_mean"] for l in batch_labels]
            std = [batch_section[l][key + "_std"] for l in batch_labels]
            ax.errorbar(xs, mean, yerr=std, marker=marker, ms=4, capsize=3,
                        lw=1.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(b) for b in xs])
        vals = [v for l in batch_labels
                for k in ("final_val_acc", "final_test_acc")
                for v in (batch_section[l][k + "_mean"] - batch_section[l][k + "_std"],
                          batch_section[l][k + "_mean"] + batch_section[l][k + "_std"])]
        ax.set_ylim(min(vals) - 0.004, max(vals) + 0.002)
        ax.set_xlabel("fitness batch size B")
        ax.set_ylabel("final accuracy")
        ax.legend()

    def width_ref_ax(ax):
        key = (1024, 1, 1024, 4096, "standardize")
        source = ref_runs if key in ref_runs else runs
        if source.get(key):
            ref = sorted(source[key], key=lambda r: r["config"]["seed"])
            steps, mean, std = traces(ref, "mode_val_acc")
            ax.plot(steps, mean, color="k", ls="--", lw=1.5,
                    label=f"h1024x1 ({num_params(1024, 1) // 1000}k)")

    def depth_trace_ax(ax):
        width_ref_ax(ax)
        for k, layers in enumerate(sorted(
                depth_section[l]["config"]["layers"] for l in depth_labels)):
            seeded = run_cell(runs, 256, layers, 1024, 4096, "standardize")
            if not seeded:
                continue
            steps, mean, std = traces(seeded, "mode_val_acc")
            colour = COLOURS(k)
            ax.plot(steps, mean, color=colour, lw=1.8,
                    label=f"h256x{layers} ({num_params(256, layers) // 1000}k)")
            ax.fill_between(steps, mean - std, mean + std, color=colour,
                            alpha=0.15, lw=0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("mode-net val accuracy")
        ax.legend(loc="lower right")

    def shaping_trace_ax(ax):
        styles = {"sh_none": ("tab:red", "none (raw advantage)"),
                  "sh_standardize": ("tab:blue", "standardize"),
                  "sh_rank": ("tab:green", "rank")}
        for k, label in enumerate(shape_labels):
            cfg = shape_section[label]["config"]
            seeded = run_cell(runs, cfg["hidden"], cfg["layers"], cfg["pop"],
                              cfg["batch"], cfg["shaping"])
            if not seeded:
                continue
            steps, mean, std = traces(seeded, "mode_val_acc")
            colour, legend = styles[label]
            ax.plot(steps, mean, color=colour, lw=1.8, label=legend)
            ax.fill_between(steps, mean - std, mean + std, color=colour,
                            alpha=0.15, lw=0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("mode-net val accuracy")
        ax.legend(loc="lower right")

    def depth_ticks():
        return [f"h256x{depth_section[l]['config']['layers']}"
                for l in depth_labels]

    def pop_trace_ax(ax, time_axis=False):
        for k, label in enumerate(pop_labels):
            cfg = pop_section[label]["config"]
            seeded = run_cell(runs, cfg["hidden"], cfg["layers"], cfg["pop"],
                              cfg["batch"], cfg["shaping"])
            if not seeded:
                continue
            steps, mean, std = traces(seeded, "mode_val_acc")
            colour = COLOURS(k)
            xs = [s / steps[-1] * pop_section[label]["wall_time_s_mean"]
                  for s in steps] if time_axis else steps
            ax.plot(xs, mean, color=colour, lw=1.8, label=f"P={cfg['pop']}")
            ax.fill_between(xs, mean - std, mean + std, color=colour,
                            alpha=0.15, lw=0)
        ax.set_ylabel("mode-net val accuracy")
        if time_axis:
            ax.set_xscale("log")
            ax.set_xlabel("estimated wall time per run (s, linear-cost model)")
        else:
            ax.set_xlabel("iteration")
        ax.legend(loc="lower right")

    # fig_pop_scaling
    if pop_labels:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        pop_ax(axes[0])
        axes[0].set_title("accuracy vs population")
        xs = [pop_section[l]["config"]["pop"] for l in pop_labels]
        walls = [pop_section[l]["wall_time_s_mean"] for l in pop_labels]
        axes[1].errorbar(xs, walls,
                         yerr=[pop_section[l]["wall_time_s_std"]
                               for l in pop_labels],
                         marker="o", ms=4, capsize=3, color=COLOURS(0), lw=1.5)
        axes[1].plot(xs, [walls[-1] * p / xs[-1] for p in xs], "--",
                     color="grey", lw=1, label="linear cost")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(xs, [str(p) for p in xs])
        axes[1].set_xlabel("population P")
        axes[1].set_ylabel("wall time per run (s)")
        axes[1].legend()
        axes[1].set_title("compute cost")
        fig.tight_layout()
        save(fig, "fig_pop_scaling")

    # fig_pop_traces
    if pop_labels:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        pop_trace_ax(axes[0])
        axes[0].set_title("population traces")
        pop_trace_ax(axes[1], time_axis=True)
        axes[1].set_title("iso-compute view")
        fig.tight_layout()
        save(fig, "fig_pop_traces")

    # fig_depth
    if depth_labels:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        depth_trace_ax(axes[0])
        axes[0].set_title("depth traces (pop 1024)")
        bar_panel(axes[1], depth_section, depth_labels, depth_ticks(),
                  "final test accuracy", COLOURS(0))
        fig.tight_layout()
        save(fig, "fig_depth")

    # fig_shaping
    if shape_labels:
        cfg = shape_section[shape_labels[0]]["config"]
        shape_title = (f"fitness shaping (h{cfg['hidden']}x{cfg['layers']}, "
                       f"pop {cfg['pop']})")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        shaping_trace_ax(axes[0])
        axes[0].set_title(shape_title)
        bar_panel(axes[1], shape_section, shape_labels,
                  ["none", "standardize", "rank"], "final test accuracy",
                  COLOURS(1))
        fig.tight_layout()
        save(fig, "fig_shaping")

    # fig_batch
    if batch_labels:
        cfg = batch_section[batch_labels[0]]["config"]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        batch_ax(ax)
        ax.set_title(f"batch size (h{cfg['hidden']}x{cfg['layers']}, "
                     f"pop {cfg['pop']})")
        fig.tight_layout()
        save(fig, "fig_batch")

    # fig_overview: 2x2 single-figure version of the available arms
    panels = []
    if pop_labels:
        panels.append(lambda ax: (pop_ax(ax), ax.set_title("population scaling")))
    if depth_labels:
        panels.append(lambda ax: bar_panel(ax, depth_section, depth_labels,
                                           depth_ticks(),
                                           "depth (h256 x L, pop 1024)",
                                           COLOURS(0)))
    if shape_labels:
        panels.append(lambda ax: bar_panel(ax, shape_section, shape_labels,
                                           ["none", "standardize", "rank"],
                                           shape_title, COLOURS(1)))
    if batch_labels:
        def batch_panel(ax):
            batch_ax(ax)
            cfg = batch_section[batch_labels[0]]["config"]
            ax.set_title(f"batch size (h{cfg['hidden']}x{cfg['layers']}, "
                         f"pop {cfg['pop']})")
        panels.append(batch_panel)
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
        for ax, panel in zip(axes.flat, panels):
            panel(ax)
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        fig.tight_layout()
        save(fig, "fig_overview")

    print("\n=== headline numbers (final val / final test, mean over seeds) ===")
    for arm, labels in (("pop", pop_labels), ("depth", depth_labels),
                        ("shape", shape_labels), ("batch", batch_labels)):
        if not labels:
            continue
        section = summary[arm]
        row = f"{arm:>6}: "
        for l in labels:
            s = section[l]
            row += (f"{l} {s['final_val_acc_mean']:.4f}/"
                    f"{s['final_test_acc_mean']:.4f}  ")
        print(row)


if __name__ == "__main__":
    main()
