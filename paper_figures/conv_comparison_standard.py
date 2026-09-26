"""Figure 1(a): the calculation process of a standard convolution.

One bank of K x K x C_in kernels produces every output channel. The parameter
counts are given in the figure caption in the manuscript rather than on the
drawing, so no formula box is placed here.

Panels (a) and (b) are drawn separately and combined into
figs/Figure_comparison_calculation.pdf.

Run:  python paper_figures/conv_comparison_standard.py
Out:  figures/Figure_comparison_calculation_a.{pdf,png}
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
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

SUB_BOX = 5.0          # side of the inner square blocks
SUB_SPACING = 0.8      # gap between them
TOTAL_HEIGHT = 33


def draw(out_dir):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 45)
    ax.set_ylim(0, TOTAL_HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc="lightblue", fontsize=12, linewidth=1.5):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc,
                                       edgecolor="black", linewidth=linewidth))
        ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize,
                ha="center", va="center", wrap=True)

    def draw_arrow(x1, y1, x2, y2, color="darkblue"):
        ax.annotate("", xy=(x2, y2), xycoords="data", xytext=(x1, y1),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", lw=1.8, color=color))

    conv_h = 9
    conv_y_center = TOTAL_HEIGHT / 2
    conv_y_start = conv_y_center - conv_h / 2

    # Input.
    input_w, input_h = 5, 2
    input_x = 12.6
    input_y_start = conv_y_center - input_h / 2
    draw_box(input_x, input_y_start, input_w, input_h,
             text="Input\n(H\u00d7W\u00d7C_in)", fc="#D1F2EB", fontsize=12)

    # Dashed frame around the convolution stage.
    conv_x, conv_w = 20, 15
    ax.add_patch(patches.Rectangle((conv_x, conv_y_start), conv_w, conv_h,
                                   facecolor="none", edgecolor="dodgerblue",
                                   linewidth=2.0, linestyle="--"))
    ax.text(conv_x + conv_w / 2, conv_y_start + conv_h - 1.4,
            "Standard Convolution", fontsize=13, color="dodgerblue",
            ha="center", va="bottom")

    margin_left = (conv_w - (2 * SUB_BOX + SUB_SPACING)) / 2
    margin_top = (conv_h - SUB_BOX) / 2
    std_conv_x = conv_x + margin_left
    std_conv_y = conv_y_start + margin_top
    draw_box(std_conv_x, std_conv_y, SUB_BOX, SUB_BOX,
             text="Conv\n(K\u00d7K\u00d7C_in\n\u00d7C_out)", fc="#E6F7FF", fontsize=11)

    bn_x = std_conv_x + SUB_BOX + SUB_SPACING
    bn_y = std_conv_y
    draw_box(bn_x, bn_y, SUB_BOX, SUB_BOX,
             text="BN + ReLU\n(H\u00d7W\u00d7C_out)", fc="#AED6F1", fontsize=11)
    draw_arrow(std_conv_x + SUB_BOX, std_conv_y + SUB_BOX / 2,
               bn_x, bn_y + SUB_BOX / 2, color="black")

    # Output.
    final_x = conv_x + conv_w + 2
    draw_box(final_x, input_y_start, input_w, input_h,
             text="Output\n(H\u00d7W\u00d7C_out)", fc="#D1F2EB", fontsize=11)

    draw_arrow(input_x + input_w, input_y_start + input_h / 2,
               std_conv_x, std_conv_y + SUB_BOX / 2, color="darkgreen")
    draw_arrow(bn_x + SUB_BOX, bn_y + SUB_BOX / 2,
               final_x, input_y_start + input_h / 2, color="black")

    plt.tight_layout()
    # Negative left margin trims the empty gutter left of the input block.
    plt.subplots_adjust(left=-0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_comparison_calculation_a"
    fig.savefig(f"{stem}.pdf", format="pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    draw(ap.parse_args().out)
