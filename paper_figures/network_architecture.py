"""Figure 3: the DS-LiteDenseNet network end to end.

Top row expands the depthwise-separable convolution and the dense block; the
lower rows lay out the stem, the four dense blocks with their transition blocks,
and the classifier.

Run:  python paper_figures/network_architecture.py
Out:  figures/Network_Architecture.{pdf,png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Embed text as TrueType rather than matplotlib's Type 3 default, and prefer a
# font from the list publishers accept. Arial is used when present, otherwise
# Helvetica, otherwise matplotlib's own DejaVu Sans.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, FancyArrowPatch

BLOCK_W, BLOCK_H = 2.2, 1.4
DETAIL_W = 4.4
DS_CONV_W = 1.8
ALPHA = 0.3

DENSE_COLOUR = "skyblue"
TRANSITION_COLOUR = "lightcoral"
CLASSIFIER_COLOUR = "gold"


def create_block(ax, x, y, width, height, label, color="lightblue", alpha=ALPHA):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=color, alpha=alpha,
                           edgecolor="black", linewidth=2.5))
    # White stroke behind the label keeps it legible over the tinted fill.
    text = ax.text(x + width / 2, y + height / 2, label, ha="center",
                   va="center", fontsize=16, fontweight="bold")
    text.set_path_effects([path_effects.Stroke(linewidth=4, foreground="white"),
                           path_effects.Normal()])


def add_arrow(ax, start, end, color="black", style="->", linewidth=2.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color,
                                 linewidth=linewidth,
                                 connectionstyle="arc3,rad=0"))


def shrink(start, end, offset=-0.05):
    """Pull an arrow head back slightly so it meets the box edge cleanly."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return end
    return (end[0] - dx / length * offset, end[1] - dy / length * offset)


def draw(out_dir):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-0.1, 15)
    ax.set_ylim(-5, 8)
    ax.axis("off")
    plt.title("DS-LiteDenseNet Network Architecture", pad=20, fontsize=24,
              fontweight="bold")

    dense_x, transition_x, pool_x = 7.5, 10.0, 12.5
    detail_y, y_base = 5.5, 2.5
    dense_centre = dense_x + BLOCK_W / 2
    detail_x = dense_centre - DETAIL_W / 2
    ds_conv_x = 0.0

    # Stem: input, strided DS convolution, max pooling.
    stem = [(0, "Input\n224\u00d7224\u00d73", "lightgreen"),
            (2.5, "DS-Conv 7\u00d77\nstride=2", "lightsalmon"),
            (5, "MaxPool 3\u00d73\nstride=2", "lightgray")]
    for x, label, colour in stem:
        create_block(ax, x, y_base, BLOCK_W, BLOCK_H, label, colour)
    for i in range(len(stem)):
        x1 = stem[i][0] + BLOCK_W
        x2 = stem[i + 1][0] if i < len(stem) - 1 else dense_x
        add_arrow(ax, (x1, y_base + BLOCK_H / 2), (x2, y_base + BLOCK_H / 2))

    # Expanded views above the main path.
    create_block(ax, detail_x, detail_y, DETAIL_W, 1.2,
                 "Dense Block Structure\nDS-Conv + Skip Connections",
                 DENSE_COLOUR)
    create_block(ax, ds_conv_x, detail_y, DS_CONV_W, 1.2, "DW Conv\n3\u00d73", "coral")
    create_block(ax, ds_conv_x + 2.2, detail_y, DS_CONV_W, 1.2, "PW Conv\n1\u00d71", "coral")
    add_arrow(ax, (ds_conv_x + DS_CONV_W, detail_y + 0.6),
              (ds_conv_x + 2.25, detail_y + 0.6))
    add_arrow(ax, (ds_conv_x + 4.0, detail_y + 0.6), (detail_x + 0.05, detail_y + 0.6),
              color="red", linewidth=3)
    for dx in (0.0, -0.2):
        add_arrow(ax, (dense_centre + dx, detail_y),
                  (dense_centre + dx, y_base + BLOCK_H - 0.05),
                  color="red", linewidth=3)
    ax.text(ds_conv_x, detail_y + 1.5, "Depthwise Separable Convolution",
            fontsize=16, fontweight="bold")
    ax.text(detail_x, detail_y + 1.5,
            "Dense Block with DS-Conv and Skip Connections",
            fontsize=16, fontweight="bold")

    # Four dense blocks stacked downwards, each followed by a transition block.
    layers = [4, 6, 8, 6]
    rows = [y_base - 2.5 * i for i in range(4)]
    for i, y in enumerate(rows):
        create_block(ax, dense_x, y, BLOCK_W, BLOCK_H,
                     f"Dense Block {i + 1}\n({layers[i]} layers)", DENSE_COLOUR)
    for y in rows[:-1]:
        create_block(ax, transition_x, y, BLOCK_W, BLOCK_H, "Transition\nBlock",
                     TRANSITION_COLOUR)

    for y in rows[:-1]:
        next_y = y - 2.5
        mid_y = next_y + BLOCK_H + 0.5
        turn_x = transition_x + BLOCK_W + 0.5
        legs = [((dense_x + BLOCK_W, y + BLOCK_H / 2), (transition_x, y + BLOCK_H / 2)),
                ((transition_x + BLOCK_W, y + BLOCK_H / 2), (turn_x, y + BLOCK_H / 2)),
                ((turn_x, y + BLOCK_H / 2), (turn_x, mid_y)),
                ((turn_x, mid_y), (dense_x + BLOCK_W / 2, mid_y)),
                ((dense_x + BLOCK_W / 2, mid_y), (dense_x + BLOCK_W / 2, next_y + BLOCK_H))]
        for start, end in legs:
            add_arrow(ax, start, shrink(start, end))

    create_block(ax, pool_x, rows[-1], BLOCK_W, BLOCK_H, "Global Pool\n& Compact Head",
                 CLASSIFIER_COLOUR)
    start = (dense_x + BLOCK_W, rows[-1] + BLOCK_H / 2)
    add_arrow(ax, start, shrink(start, (pool_x, rows[-1] + BLOCK_H / 2)))

    legend = [("lightgreen", "Input Layer"), ("lightsalmon", "DS Convolution"),
              (DENSE_COLOUR, "Dense Block"), (TRANSITION_COLOUR, "Transition Block"),
              (CLASSIFIER_COLOUR, "Classification")]
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=ALPHA, label=l)
               for c, l in legend]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none",
                             label="* All Dense Blocks share identical internal structure"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=5, fontsize=14)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem_path = out_dir / "Network_Architecture"
    fig.savefig(f"{stem_path}.pdf", bbox_inches="tight", pad_inches=0.3)
    fig.savefig(f"{stem_path}.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {stem_path}.pdf and {stem_path}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    draw(ap.parse_args().out)
