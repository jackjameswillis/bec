"""Render the STE / ES / GA comparison diagram (figures/Wr) as a vector figure.

Outputs: figures/Wr.pdf (vector, for papers) and figures/Wr.png (high-dpi raster).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "text.color": "#1a1a1a",
})

NODE_R = 0.024          # node half-size (data units) — smaller = more breathing room
NODE_PAD = 0.006        # clearance from node edge — arrows end right at box edges
ARROW_KW = dict(arrowstyle="-|>", lw=1.7, color="#1a1a1a", mutation_scale=18)

NODE_FC = "#f6f6f6"
NODE_EC = "#1a1a1a"
NODE_LW = 1.0
LABEL_COLOR = "#2b2b2b"

NODE_FS = 17            # symbol font size inside boxes
LABEL_FS = 15.5         # plain-text label font size
TITLE_FS = 18.5         # panel title font size


def wrap2(text):
    """Split any multi-word label onto two lines (balanced, unless pre-split)."""
    if "\n" in text:
        return text
    words = text.split()
    if len(words) <= 1:
        return text
    if len(words) == 2:
        return words[0] + "\n" + words[1]
    k = math.ceil(len(words) / 2)
    return " ".join(words[:k]) + "\n" + " ".join(words[k:])


def boundary_point(c, t):
    cx, cy = c
    tx, ty = t
    dx, dy = tx - cx, ty - cy
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    scale = NODE_R / max(abs(dx), abs(dy)) if max(abs(dx), abs(dy)) > 0 else NODE_R
    return cx + dx * (scale + NODE_PAD), cy + dy * (scale + NODE_PAD)


def outward_label_pos(c1, c2, d=0.085):
    """Label placed just outside the panel triangle, near the c1->c2 line."""
    mx, my = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
    cx, cy = 0.5, 0.5  # panel centre
    dx, dy = mx - cx, my - cy
    n = math.hypot(dx, dy) or 1.0
    return mx + dx / n * d, my + dy / n * d


def draw_node(ax, c, text):
    cx, cy = c
    box = FancyBboxPatch(
        (cx - NODE_R, cy - NODE_R), 2 * NODE_R, 2 * NODE_R,
        boxstyle="square,pad=0.14",
        transform=ax.transData,
        fc=NODE_FC, ec=NODE_EC, lw=NODE_LW, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=NODE_FS,
            zorder=4, family="serif")


def draw_label(ax, pos, text, ha="center", va="center", offset=(0.0, 0.0)):
    x, y = pos
    ax.text(x + offset[0], y + offset[1], wrap2(text), ha=ha, va=va,
            fontsize=LABEL_FS, color=LABEL_COLOR, zorder=4, family="serif")


def draw_arrow(ax, src, dst, rad=0.0):
    p1 = boundary_point(src, dst)  # start at src edge
    p2 = boundary_point(dst, src)  # end at dst edge

    # end arrowhead at dst
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(ARROW_KW, connectionstyle=f"arc3,rad={rad}"),
        annotation_clip=False,
        zorder=1,
    )

    # mid-line arrowhead pointing in flow direction
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    mlen = 0.035
    ax.annotate(
        "", xy=(mx + dx * mlen, my + dy * mlen),
        xytext=(mx - dx * mlen, my - dy * mlen),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#1a1a1a",
                        mutation_scale=13),
        annotation_clip=False,
        zorder=1,
    )


# Each panel: title, top node (x, y, symbol, label), bottom nodes, clockwise
# flow [(src, dst, label, ha, va, rad), ...], below-panel label.
PANELS = [
    dict(
        title="STE",
        top=(0.50, 0.72, r"$W_r$", "surrogate weights"),
        bottom=dict(
            left=(0.20, 0.30, r"$g$", "gradient"),
            right=(0.80, 0.30, r"$W_b$", "binary weights"),
        ),
        flow=[
            # clockwise loop: top -> right -> left -> top
            ("top", "right", "quantize", "left", "center", 0.0),
            ("right", "left", None, "center", "top", 0.0),
            ("left", "top", "gradient descent", "right", "center", 0.0),
        ],
        below=("backpropagation\nvia STE", (0.50, 0.06)),
    ),
    dict(
        title="ES",
        top=(0.50, 0.72, r"$\theta$", "search distribution"),
        bottom=dict(
            left=(0.20, 0.30, r"$\hat{g}$", "gradient approximation"),
            right=(0.80, 0.30, r"$P$", "pseudo population"),
        ),
        flow=[
            ("top", "right", "sample parameters", "left", "center", 0.0),
            ("right", "left", None, "center", "top", 0.0),
            ("left", "top", "gradient ascent", "right", "center", 0.0),
        ],
        below=("fitness evaluation", (0.50, 0.06)),
    ),
    dict(
        title="GA",
        top=(0.50, 0.72, r"$P$", "population"),
        bottom=dict(
            left=(0.20, 0.30, r"$L'$", "child"),
            right=(0.80, 0.30, r"$W,\ L$", "winner and loser"),
        ),
        flow=[
            ("top", "right", "tournament selection", "left", "center", 0.0),
            ("right", "left", None, "center", "top", 0.0),
            ("left", "top", "loser replacement", "right", "center", 0.0),
        ],
        below=("crossover and mutation", (0.50, 0.06)),
    ),
]


def draw_panel(ax, panel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, 0.965, panel["title"], ha="center", va="bottom",
            fontsize=TITLE_FS, fontweight="bold", family="serif")

    tx, ty, ttext, tlabel = panel["top"]
    bx_l, by_l, ltext, llabel = panel["bottom"]["left"]
    bx_r, by_r, rtext, rlabel = panel["bottom"]["right"]

    centers = {
        "top": (tx, ty),
        "left": (bx_l, by_l),
        "right": (bx_r, by_r),
    }

    # top node + label centred inside the box, above the symbol
    draw_node(ax, centers["top"], ttext)
    draw_label(ax, (tx, ty + 0.085), tlabel, "center", "center")

    # bottom nodes + labels below
    draw_node(ax, centers["left"], ltext)
    draw_label(ax, (bx_l, by_l - NODE_R - 0.045), llabel, "center", "top")
    draw_node(ax, centers["right"], rtext)
    draw_label(ax, (bx_r, by_r - NODE_R - 0.045), rlabel, "center", "top")

    # arrows + side labels
    for src, dst, label, ha, va, rad in panel["flow"]:
        draw_arrow(ax, centers[src], centers[dst], rad=rad)
        if label is not None:
            lx, ly = outward_label_pos(centers[src], centers[dst], d=0.085)
            draw_label(ax, (lx, ly), label, ha, va)

    # below-panel label (e.g. backpropagation)
    below_text, (bx, by) = panel["below"]
    draw_label(ax, (bx, by), below_text, "center", "top")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.4), constrained_layout=True)
    for ax, panel in zip(axes, PANELS):
        draw_panel(ax, panel)

    pdf = OUT_DIR / "Wr.pdf"
    png = OUT_DIR / "Wr.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
