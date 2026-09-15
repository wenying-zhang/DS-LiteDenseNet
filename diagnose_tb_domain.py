"""Supplementary Fig. S3: separate input-level differences from genuine domain shift on the TB task.

Distinguishes two explanations for the drop between Shenzhen and Montgomery: an
input-level mismatch in bit depth, intensity or contrast, which preprocessing
can correct, and a difference in the pathology itself, which it cannot.

Pixel bit depth is read directly from disk, independent of datasets.py, so the
diagnosis holds whether or not the preprocessing is enabled. Intensity summaries
are printed for three stages of the pipeline and a side-by-side panel is saved.

Run:  python diagnose_tb_domain.py
Output: figures/diag_tb_domain_panel.png and a printed table.
"""
import numpy as np
import pandas as pd
from PIL import Image
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

import config as C
from datasets import load_image, build_transforms


def _sample(df, n, seed=0):
    return df if len(df) <= n else df.sample(n, random_state=seed)


def summarize(df, name, n=40):
    """Report bit depth + intensity at raw / after-convert / after-transform."""
    sub = _sample(df, n)
    if len(sub) == 0:
        print(f"\n===== {name} =====\n  no images found; is this corpus present "
              f"and are its paths resolving? (see data_manifests.py)")
        return sub
    modes, raw, conv, tens = {}, [], [], []
    tfm = build_transforms(train=False)
    for _, r in sub.iterrows():
        pil = Image.open(r["path"])
        modes[pil.mode] = modes.get(pil.mode, 0) + 1
        a = np.asarray(pil).astype(np.float32)
        raw.append((a.min(), a.max(), a.mean()))
        g = np.asarray(load_image(r["path"], r["file_type"]).convert("L")).astype(np.float32)
        conv.append((g.min(), g.max(), g.mean()))
        t = tfm(load_image(r["path"], r["file_type"]))
        tens.append((float(t.min()), float(t.max()), float(t.mean())))
    raw, conv, tens = np.array(raw), np.array(conv), np.array(tens)
    print(f"\n===== {name}  (n={len(sub)}) =====")
    print(f"  PIL modes           : {modes}")
    print(f"  RAW pixel   min/max/mean : {raw[:,0].mean():8.1f} /{raw[:,1].mean():9.1f} /{raw[:,2].mean():8.1f}")
    print(f"  after convert('L')       : {conv[:,0].mean():8.1f} /{conv[:,1].mean():9.1f} /{conv[:,2].mean():8.1f}")
    print(f"  after eval transform     : {tens[:,0].mean():8.2f} /{tens[:,1].mean():9.2f} /{tens[:,2].mean():8.2f}")
    return sub


def panel(shen_df, mont_df, k=4):
    # A narrow first column carries the row label so that every image axis keeps
    # the same size and nothing is drawn over a radiograph.
    fig, axes = plt.subplots(
        2, k + 1, figsize=(3 * k + 0.8, 6.2), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[0.10] + [1.0] * k))
    for row, (df, title) in enumerate([(shen_df, "Shenzhen"), (mont_df, "Montgomery")]):
        lab = axes[row, 0]
        lab.axis("off")
        lab.text(0.5, 0.5, title, transform=lab.transAxes, fontsize=18,
                 rotation=90, va="center", ha="center")
        # Blank every image axis first. A corpus with fewer than k images would
        # otherwise leave the unused ones drawn as empty framed plots with tick
        # labels, in the middle of a figure of radiographs.
        for ax in axes[row, 1:]:
            ax.axis("off")
        for i, (_, r) in enumerate(_sample(df, k, seed=1).iterrows()):
            g = np.asarray(load_image(r["path"], r["file_type"]).convert("L"))
            ax = axes[row, i + 1]
            ax.imshow(g, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
    stem = C.FIG_DIR / "diag_tb_domain_panel"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"\nSaved visual panel -> {stem}.pdf (+ .png)")


def main():
    tb_int = pd.read_csv(C.MANIFEST_DIR / "tb_internal.csv")
    shen = tb_int[(tb_int.split == "test") & (tb_int.source == "shenzhen")]
    if len(shen) == 0:
        shen = tb_int[tb_int.source == "shenzhen"]
    mont = pd.read_csv(C.MANIFEST_DIR / "tb_external.csv")
    mont = mont[mont.source == "montgomery"]

    shen_s = summarize(shen, "SHENZHEN (internal test)")
    mont_s = summarize(mont, "MONTGOMERY (external)")
    panel(shen_s, mont_s)

    print("\n--- interpreting the panel ---")
    print("* Montgomery mode is I / I;16 / F, or RAW max >> 255, while Shenzhen is")
    print("  'L' with max ~255  ->  a naive 8-bit conversion clips the high")
    print("  bit depth; the windowing in datasets.py handles it.")
    print("* Both are mode 'L', max ~255, but the panel shows clearly different")
    print("  brightness/contrast  ->  real intensity shift, which a uniform")
    print("  normalisation such as CLAHE addresses.")
    print("* Images look alike  ->  the remaining gap is a pathology or")
    print("  population shift rather than an input-level one.")


if __name__ == "__main__":
    main()
