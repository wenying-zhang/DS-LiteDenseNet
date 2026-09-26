"""Figure 7: locked-test AUROC against the fraction of pneumonia training labels.

Reads the per-fraction summaries written by run_ablation.py rather than carrying
transcribed numbers, so the figure cannot drift away from the results it plots.
Each fraction needs a prior run:

    for f in 0.05 0.1 0.25 0.5 1.0; do
        python run_ablation.py --task pneumonia --label_fraction $f
    done

Run:  python paper_figures/label_efficiency.py
Out:  figures/label_efficiency.{pdf,png}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]

# Result-file key -> label used in the legend.
MODELS = {
    "DenseNet121":    "DenseNet-121",
    "DenseNet121DS":  "DenseNet-121-DS",
    "LiteDenseNet":   "LiteDenseNet",
    "DSLiteDenseNet": "DS-LiteDenseNet",
    "MobileNetV2":    "MobileNetV2",
    "ShuffleNetV2":   "ShuffleNetV2",
}

STYLES = {
    "DenseNet-121":    dict(color="#444444", marker="o", lw=1.6, ls="--"),
    "DenseNet-121-DS": dict(color="#8c8c8c", marker="^", lw=1.2, ls=":"),
    "LiteDenseNet":    dict(color="#5b9bd5", marker="s", lw=1.2, ls=":"),
    "DS-LiteDenseNet": dict(color="#c00000", marker="D", lw=2.6, ls="-"),
    "MobileNetV2":     dict(color="#70ad47", marker="v", lw=1.2, ls="-."),
    "ShuffleNetV2":    dict(color="#ed7d31", marker="P", lw=1.2, ls="-."),
}


def tag_for(fraction, task="pneumonia"):
    """Match the tag run_ablation.py builds.

    That tag interpolates the float itself, so 1.0 becomes 'lf1.0' and not
    'lf1'; formatting the value here would silently look for the wrong file.
    """
    return f"{task}_lf{float(fraction)}_scratch"


def read_curves(results_dir, task):
    """Return {display name: (means, stds)} indexed by FRACTIONS."""
    means = {name: [] for name in MODELS.values()}
    stds = {name: [] for name in MODELS.values()}
    for fraction in FRACTIONS:
        path = results_dir / f"ablation_{tag_for(fraction, task)}_summary.csv"
        if not path.exists():
            raise SystemExit(
                f"missing {path}\n"
                f"run: python run_ablation.py --task {task} "
                f"--label_fraction {fraction:g}")
        df = pd.read_csv(path, index_col=0)
        for key, name in MODELS.items():
            if key not in df.index:
                raise SystemExit(f"{path} has no row for {key}")
            means[name].append(float(df.loc[key, "auroc_mean"]))
            stds[name].append(float(df.loc[key, "auroc_std"]))
    return {n: (np.array(means[n]), np.array(stds[n])) for n in MODELS.values()}


def report(curves):
    header = "  ".join(f"{int(f * 100):>3d}%" for f in FRACTIONS)
    print(f"Locked-test AUROC, mean over seeds:\n{'model':18s} {header}")
    for name, (mu, _) in curves.items():
        print(f"{name:18s} " + "  ".join(f"{v:.3f}" for v in mu))
    ours = curves["DS-LiteDenseNet"][0][0]
    print(f"\nAt 5%: DS-LiteDenseNet minus MobileNetV2 = "
          f"{ours - curves['MobileNetV2'][0][0]:+.3f} AUROC, "
          f"minus ShuffleNetV2 = {ours - curves['ShuffleNetV2'][0][0]:+.3f}.")


def draw(curves, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, (mu, sd) in curves.items():
        ax.errorbar(FRACTIONS, mu, yerr=sd, label=name, capsize=2.5,
                    markersize=5.5,
                    zorder=5 if name == "DS-LiteDenseNet" else 3, **STYLES[name])
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in FRACTIONS])
    ax.minorticks_off()
    ax.set_xlabel("Fraction of pneumonia training labels")
    ax.set_ylabel("Locked-test AUROC")
    lo = min(mu.min() - sd.max() for mu, sd in curves.values())
    hi = max(mu.max() + sd.max() for mu, sd in curves.values())
    ax.set_ylim(lo - 0.002, hi + 0.002)
    ax.grid(True, which="major", ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "label_efficiency"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--out", type=Path, default=root / "figures")
    ap.add_argument("--task", default="pneumonia")
    args = ap.parse_args()
    data = read_curves(args.results, args.task)
    report(data)
    draw(data, args.out)
