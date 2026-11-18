"""Render the STE / ES comparison diagram (figures/evostar_peg) as a vector figure.

Same as figures/Wr but without the GA loop: only the STE and ES panels.

Outputs: figures/evostar_peg.pdf (vector, for papers) and figures/evostar_peg.png (high-dpi raster).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_wr_figure import PANELS, draw_panel

OUT_DIR = REPO / "figures"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 5.4), constrained_layout=True)
    for ax, panel in zip(axes, PANELS[:2]):
        draw_panel(ax, panel)

    pdf = OUT_DIR / "evostar_peg.pdf"
    png = OUT_DIR / "evostar_peg.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
