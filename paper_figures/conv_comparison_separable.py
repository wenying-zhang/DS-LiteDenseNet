"""Figure 1(b): the calculation process of a depthwise-separable convolution.

A per-channel K x K spatial filter is applied to each input channel and a 1 x 1
pointwise convolution then mixes the channels. Parameter counts are given in the
figure caption in the manuscript rather than on the drawing, so no formula box
is placed here.

Panels (a) and (b) are drawn separately and combined into
figs/Figure_comparison_calculation.pdf.

Run:  python paper_figures/conv_comparison_separable.py
Out:  figures/Figure_comparison_calculation_b.{pdf,png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

# Embed text as TrueType rather than matplotlib's Type 3 default, and prefer a
# font from the list publishers accept. Arial is used when present, otherwise
# Helvetica, otherwise matplotlib's own DejaVu Sans.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

SUB_BOX = 5.0
SUB_SPACING = 0.8


def draw(out_dir):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 33)
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

    def draw_folded(points, color="darkblue", arrow_on_last=True):
        """Right-angled polyline; only the final segment carries an arrow head."""
        for i in range(len(points) - 1):
            (x1, y1), (x2, y2) = points[i], points[i + 1]
            if arrow_on_last and i == len(points) - 2:
                ax.annotate("", xy=(x2, y2), xycoords="data", xytext=(x1, y1),
                            textcoords="data",
                            arrowprops=dict(arrowstyle="->", lw=1.8, color=color))
            else:
                ax.add_line(mlines.Line2D([x1, x2], [y1, y2], lw=1.8, color=color))

    def draw_channel(label, x, y, w, h):
        """One dashed channel box holding a depthwise filter and its BN+ReLU."""
        margin_left = (w - (2 * SUB_BOX + SUB_SPACING)) / 2
        margin_top = (h - SUB_BOX) / 2
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none",
                                       edgecolor="dodgerblue", linewidth=2.0,
                                       linestyle="--"))
        ax.text(x + w / 2, y + h - 1.4, label, fontsize=13, color="dodgerblue",
                ha="center", va="bottom")

        dw_x, dw_y = x + margin_left, y + margin_top
        draw_box(dw_x, dw_y, SUB_BOX, SUB_BOX, text="DWConv\n(K\u00d7K, g=in)",
                 fc="#E6F7FF", fontsize=11)
        bn_x, bn_y = dw_x + SUB_BOX + SUB_SPACING, dw_y
        draw_box(bn_x, bn_y, SUB_BOX, SUB_BOX, text="BN + ReLU\n(H\u00d7W\u00d71)",
                 fc="#AED6F1", fontsize=11)
        draw_arrow(dw_x + SUB_BOX, dw_y + SUB_BOX / 2, bn_x, bn_y + SUB_BOX / 2,
                   color="black")
        return dw_x, dw_y + SUB_BOX / 2, bn_x + SUB_BOX / 2, bn_y + SUB_BOX / 2

    # Three channel boxes, centred horizontally between the input and the concat.
    concat_x = 27
    input_x, input_w, input_h = 5, 5, 2
    input_right = input_x + input_w
    cw, ch = 12, 9
    c_x = (input_right + concat_x) / 2 - cw / 2
    rows = [21, 11, 1]

    # Align the input with the middle channel's depthwise block.
    mid_dw_centre_y = rows[1] + (ch - SUB_BOX) / 2 + SUB_BOX / 2
    input_y = mid_dw_centre_y - input_h / 2
    draw_box(input_x, input_y, input_w, input_h, text="Input\n(H\u00d7W\u00d7C_in)",
             fc="#D1F2EB", fontsize=12)
    input_centre_y = input_y + input_h / 2

    dw_left = c_x + (cw - (2 * SUB_BOX + SUB_SPACING)) / 2
    pivot_x = (input_right + dw_left) / 2

    bn_anchors = []
    for i, y in enumerate(rows):
        mid_y = y + ch / 2
        draw_folded([(input_right, input_centre_y), (pivot_x, input_centre_y),
                     (pivot_x, mid_y)], color="darkgreen", arrow_on_last=False)
        dw_left_x, dw_left_y, bn_cx, bn_cy = draw_channel(
            f"Channel #{i + 1}", c_x, y, cw, ch)
        draw_arrow(pivot_x, mid_y, dw_left_x, dw_left_y, color="darkgreen")
        bn_anchors.append((bn_cx + SUB_BOX / 2, bn_cy))

    # Concatenation, then the pointwise stage.
    concat_y = mid_dw_centre_y - 1.0
    draw_box(concat_x, concat_y, 5, 2, text="Concat\n(H\u00d7W\u00d7C_in)",
             fc="#FDEDEC", fontsize=11)
    pivot_right = concat_x - 1.5
    concat_target_y = concat_y + 1.0
    for bn_cx, bn_cy in bn_anchors:
        draw_folded([(bn_cx, bn_cy), (pivot_right, bn_cy),
                     (pivot_right, concat_target_y), (concat_x, concat_target_y)],
                    color="darkgreen", arrow_on_last=True)

    combine_x, combine_w, combine_h = 35 - 0.8, 12, 9
    combine_bottom = rows[1]
    combine_centre_y = combine_bottom + combine_h / 2
    ax.add_patch(patches.Rectangle((combine_x, combine_bottom), combine_w,
                                   combine_h, facecolor="none",
                                   edgecolor="dodgerblue", linewidth=2.0,
                                   linestyle="--"))
    ax.text(combine_x + combine_w / 2, combine_bottom + combine_h - 1.4,
            "Combine channels", fontsize=13, color="dodgerblue", ha="center",
            va="bottom")
    draw_folded([(concat_x + 5, concat_y + 1.0), (concat_x + 7, concat_y + 1.0),
                 (concat_x + 7, combine_centre_y), (combine_x, combine_centre_y)],
                color="darkgreen", arrow_on_last=True)

    margin_left = (combine_w - (2 * SUB_BOX + SUB_SPACING)) / 2
    margin_top = (combine_h - SUB_BOX) / 2
    pw_x = combine_x + margin_left
    pw_y = combine_bottom + margin_top
    draw_box(pw_x, pw_y, SUB_BOX, SUB_BOX, text="PWConv\n(1\u00d71)",
             fc="#E6F7FF", fontsize=11)
    bn_x, bn_y = pw_x + SUB_BOX + SUB_SPACING, pw_y
    draw_box(bn_x, bn_y, SUB_BOX, SUB_BOX, text="BN\n(H\u00d7W\u00d7C_out)",
             fc="#FFF5E6", fontsize=11)
    draw_arrow(pw_x + SUB_BOX, pw_y + SUB_BOX / 2, bn_x, bn_y + SUB_BOX / 2,
               color="black")

    final_x = combine_x + combine_w + 2.0
    final_y = combine_bottom + (combine_h - input_h) / 2
    draw_box(final_x, final_y, input_w, input_h, text="Final Output\n(H\u00d7W\u00d7C_out)",
             fc="#D1F2EB", fontsize=11)
    draw_arrow(bn_x + SUB_BOX, bn_y + SUB_BOX / 2, final_x, final_y + input_h / 2,
               color="black")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_comparison_calculation_b"
    fig.savefig(f"{stem}.pdf", format="pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    draw(ap.parse_args().out)
