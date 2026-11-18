import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bes.mlp import weight_shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "results/es_h256x2_he_relu_p2048_lr0.048828125_shrank_seed42_t0_diag5000_theta.npz"))
    ap.add_argument("--iters", default="100,500,5000")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "figures/theta_expectations_layers_h256x2_p2048_seed42"))
    args = ap.parse_args()

    z = np.load(args.npz)
    ks = sorted([k for k in z.keys() if k.startswith("theta_")], key=lambda s: int(s.split("it")[-1]))
    avail = {int(k.split("it")[-1]): k for k in ks}
    iters = [int(s) for s in args.iters.split(",")]
    missing = [i for i in iters if i not in avail]
    if missing:
        raise SystemExit(f"iterations {missing} not in npz (available: {sorted(avail)})")

    shapes = weight_shapes(hidden_units=(args.hidden,) * args.layers)
    bins = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    hist = {}
    scale = {}
    for li, (in_dim, units) in enumerate(shapes):
        lo = sum(a * b for a, b in shapes[:li])
        hi = lo + in_dim * units
        row = {}
        ymax = 0.0
        ymin = np.inf
        for it in iters:
            theta = z[avail[it]][lo:hi].astype(np.float64)
            p = 1.0 / (1.0 + np.exp(-theta))
            h, _ = np.histogram(p, bins=bins)
            row[it] = h
            ymax = max(ymax, h.max())
            ymin = min(ymin, h[h > 0].min())
        hist[li] = row
        scale[li] = (0.5 * ymin, ymax * 3,
                     [t for t in [1, 10, 100, 1e3, 1e4, 1e5, 1e6] if 0.5 * ymin <= t <= ymax * 3])

    nl = len(shapes)
    fig, axes = plt.subplots(nl, len(iters), figsize=(5.0 * len(iters), 3.2 * nl), squeeze=False)
    for ci, it in enumerate(iters):
        axes[0][ci].set_title(f"iteration {it}", fontsize=13)
        for li in range(nl):
            ax = axes[li][ci]
            ybot, ytop, yticks = scale[li]
            ax.bar(centers, hist[li][it], width=width, color="tab:blue", edgecolor="none")
            ax.set_yscale("log")
            ax.set_ylim(ybot, ytop)
            ax.set_yticks(yticks)
            ax.set_xlabel("Bernoulli expectation")
            if ci == 0:
                ax.set_ylabel(f"layer {li + 1}\nbit count (log)")
    fig.suptitle("later layers converge faster", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(args.out + ext, dpi=200)
    print(f"wrote {args.out}.pdf/.png (iters {iters}, layers {nl}: {shapes}, {args.bins} bins)")


if __name__ == "__main__":
    main()
