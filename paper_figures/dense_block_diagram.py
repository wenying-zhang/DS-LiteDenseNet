"""Figure 2: dense-layer structure of DenseNet-121 and DS-LiteDenseNet.

Draws the two dense layers side by side so the only difference between them,
the 3x3 standard convolution against the depthwise-separable pair, is visible in
one glance. Both rows share an x grid so the Concat blocks line up.

Run:  python paper_figures/dense_block_diagram.py
Out:  figures/Figure_Dense_Block_Structure.{pdf,png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Embed text as TrueType rather than matplotlib's Type 3 default, and prefer a
# font from the list publishers accept. Arial is used when present, otherwise
# Helvetica, otherwise matplotlib's own DejaVu Sans.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BOX_W, BOX_H = 2.4, 2.5
SPACING = 0.5
X_START = 1.5
Y_BASE = 3.5
SKIP_COLOURS = ["darkgreen", "forestgreen", "seagreen",
                "mediumseagreen", "mediumaquamarine"]


def draw_box(ax, x, y, w, h, text, fc="lightblue", fontsize=13):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                                   linewidth=1.5, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize,
            ha="center", va="center", wrap=True, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xycoords="data", xytext=(x1, y1),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color), zorder=2)


def draw_skips(ax, starts, x_end):
    """Curved skip connections from each block to the concatenation."""
    for i, x0 in enumerate(starts):
        ax.add_patch(patches.FancyArrowPatch(
            (x0 + BOX_W / 2, Y_BASE), (x_end + BOX_W / 2, Y_BASE),
            connectionstyle=f"arc3,rad={0.15 + i * 0.08}", arrowstyle="->",
            color=SKIP_COLOURS[i], linewidth=1.5, zorder=1))


def draw(out_dir):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8.0),
        gridspec_kw={"hspace": 0.1, "top": 0.95, "bottom": 0.05})

    for ax in (ax1, ax2):
        ax.set_xlim(0, 19.5)
        ax.set_ylim(-1.0, 7.0)
        ax.axis("off")

    x = [X_START + i * (BOX_W + SPACING) for i in range(6)]
    centre = (x[0] + x[5] + BOX_W) / 2
    title_y = Y_BASE + BOX_H + 0.4

    ax1.text(0.2, Y_BASE + BOX_H / 2, "(a)", fontsize=24, va="center",
             fontweight="bold")
    ax2.text(0.2, Y_BASE + BOX_H / 2, "(b)", fontsize=24, va="center",
             fontweight="bold")
    ax1.text(centre, title_y, "DenseNet-121 Dense Layer Structure",
             fontsize=20, ha="center", va="bottom")
    ax2.text(centre, title_y, "DS-LiteDenseNet Dense Layer Structure",
             fontsize=20, ha="center", va="bottom")

    # Both panels write the growth rate as g rather than growth_rate. The two
    # boxes carry different quantities that happen to be numerically equal --
    # the bottleneck's output channel count and the depthwise convolution's
    # group count -- so the symbol has to match across them or a reader will
    # think one of the two is a different parameter. growth_rate spelled out
    # does not fit in the groups box, so g is used in both. The caption of
    # Fig. 2 in the manuscript defines it.

    # (a) DenseNet-121: a single 3x3 standard convolution after the bottleneck.
    draw_box(ax1, x[0], Y_BASE, BOX_W, BOX_H, "Input\n(H\u00d7W\u00d7C)", fc="#D1F2EB")
    draw_box(ax1, x[1], Y_BASE, BOX_W, BOX_H, "BN + ReLU", fc="#AED6F1")
    draw_box(ax1, x[2], Y_BASE, BOX_W, BOX_H,
             "1\u00d71 Conv\n(4\u00d7g channels)\n+ BN + ReLU", fc="#E6F7FF", fontsize=12)
    draw_box(ax1, x[3], Y_BASE, BOX_W, BOX_H, "3\u00d73 Conv\n(Standard)", fc="#F5EEF8")
    # Slot x[4] is left empty so both rows end at the same Concat column.
    draw_box(ax1, x[5], Y_BASE, BOX_W, BOX_H, "Concat\n(Feature\nFusion)", fc="#FDEDEC")
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        draw_arrow(ax1, x[a] + BOX_W, Y_BASE + BOX_H / 2, x[b], Y_BASE + BOX_H / 2)
    draw_arrow(ax1, x[3] + BOX_W, Y_BASE + BOX_H / 2, x[5], Y_BASE + BOX_H / 2)
    draw_skips(ax1, [x[0]], x[5])
    ax1.text(centre, 0.0, "Skip Connection (Input Concatenated with Output)",
             fontsize=16, color="darkgreen", ha="center", va="top")

    # (b) DS-LiteDenseNet: the 3x3 is factorised into depthwise and pointwise.
    draw_box(ax2, x[0], Y_BASE, BOX_W, BOX_H, "Input\n(H\u00d7W\u00d7C)", fc="#D1F2EB")
    draw_box(ax2, x[1], Y_BASE, BOX_W, BOX_H, "BN + ReLU", fc="#AED6F1")
    draw_box(ax2, x[2], Y_BASE, BOX_W, BOX_H,
             "1\u00d71 Conv\n(4\u00d7g channels)\n+ BN + ReLU", fc="#E6F7FF", fontsize=12)
    draw_box(ax2, x[3], Y_BASE, BOX_W, BOX_H,
             "3\u00d73 DWConv\n+ BN + ReLU\n(groups=4\u00d7g)", fc="#F5EEF8", fontsize=12)
    draw_box(ax2, x[4], Y_BASE, BOX_W, BOX_H, "1\u00d71 PWConv\n+ BN", fc="#AED6F1")
    draw_box(ax2, x[5], Y_BASE, BOX_W, BOX_H, "Concat\n(Feature\nFusion)", fc="#FDEDEC")
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        draw_arrow(ax2, x[a] + BOX_W, Y_BASE + BOX_H / 2, x[b], Y_BASE + BOX_H / 2)
    draw_skips(ax2, [x[0]], x[5])
    ax2.text(centre, 0.0, "Skip Connection (Input Concatenated with Output)",
             fontsize=16, color="darkgreen", ha="center", va="top")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_Dense_Block_Structure"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    draw(ap.parse_args().out)
