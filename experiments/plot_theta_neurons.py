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
    ap.add_argument("--cmap", default="gray", help="matplotlib colormap for the imshows")
    ap.add_argument("--layer", type=int, default=0, help="0-based index into weight_shapes")
    ap.add_argument("--neurons", default="random", help="comma list or 'random'")
    ap.add_argument("--pick-seed", type=int, default=42, help="seed for random neuron choice")
    ap.add_argument("--out", default=str(ROOT / "figures/theta_neurons_h256x2_p2048_seed42"))
    args = ap.parse_args()

    z = np.load(args.npz)
    ks = sorted([k for k in z.keys() if k.startswith("theta_")], key=lambda s: int(s.split("it")[-1]))
    avail = {int(k.split("it")[-1]): k for k in ks}
    iters = [int(s) for s in args.iters.split(",")]
    missing = [i for i in iters if i not in avail]
    if missing:
        raise SystemExit(f"iterations {missing} not in npz (available: {sorted(avail)})")

    shapes = weight_shapes(hidden_units=(args.hidden,) * args.layers)
    in_dim, units = shapes[args.layer]
    offset = sum(a * b for a, b in shapes[:args.layer])
    if args.neurons == "random":
        rng = np.random.default_rng(args.pick_seed)
        neurons = sorted(rng.choice(units, size=min(9, units),
                                    replace=False).tolist())
    else:
        neurons = [int(s) for s in args.neurons.split(",")]

    side = int(np.sqrt(in_dim))
    assert side * side == in_dim, f"in_dim {in_dim} is not a perfect square; cannot reshape to 2D"

    mats = []
    for it in iters:
        theta = z[avail[it]][offset:offset + in_dim * units].reshape(units, in_dim)
        mats.append([theta[i].reshape(side, -1) for i in neurons])

    vmin = min(m.min() for row in mats for m in row)
    vmax = max(m.max() for row in mats for m in row)

    n = len(iters)
    nr = int(np.ceil(np.sqrt(len(neurons))))
    nc = int(np.ceil(len(neurons) / nr))
    fig = plt.figure(figsize=(2.2 * nr * n, 2.2 * nr))
    outer = fig.add_gridspec(1, n, wspace=0.12, top=0.82, bottom=0.02, left=0.01, right=0.98)
    axes = []
    for ci in range(n):
        inner = outer[ci].subgridspec(nr, nc, wspace=0.01, hspace=0.02)
        axes.append([[fig.add_subplot(inner[r, c]) for c in range(nc)] for r in range(nr)])
    for ci, it in enumerate(iters):
        for k, idx in enumerate(neurons):
            r, c = divmod(k, nc)
            ax = axes[ci][r][c]
            ax.imshow(mats[ci][k], cmap=args.cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"neuron {idx}", fontsize=8)
        axes[ci][0][nc // 2].annotate(f"iteration {it}", xy=(0.5, 1.35), xycoords="axes fraction",
                                      ha="center", fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=[ax for group in axes for row in group for ax in row],
                        fraction=0.02, pad=0.01)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0f}" if abs(vmax) >= 4 else f"{t:.1f}" for t in ticks])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(args.out + ext, dpi=200)
    print(f"wrote {args.out}.pdf/.png (iters {iters}, layer {args.layer}: {in_dim}->{units}, neurons {neurons})")


if __name__ == "__main__":
    main()
