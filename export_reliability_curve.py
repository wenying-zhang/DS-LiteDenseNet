"""Export a single reliability curve as vector coordinates and a minimal PDF.

Reads the per-sample probabilities written by run_ablation.py, pools the seeds
and bins by confidence, following Guo et al. (2017): confidence is the
probability assigned to the predicted class, max(p, 1-p). The output is a small
standalone panel rather than the multi-model figure produced by
plot_reliability.py. The CSV holds the exact bin coordinates, so the curve can
be redrawn faithfully in a vector editor without re-deriving it.

Usage:
    python export_reliability_curve.py
    python export_reliability_curve.py --model DenseNet121

Outputs, in the working directory:
    reliability_curve_<model>_<task>.csv       bin confidence, accuracy, count
    reliability_curve_<model>_<task>.pdf       vector panel
"""
import argparse, glob, os
import numpy as np
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

try:
    import config as C
    PROB_DIR = str(C.RESULTS_DIR)
    _DEFAULT_SEEDS = list(C.SEEDS)
except Exception:
    PROB_DIR = "results"        # fallback when run outside the project tree
    _DEFAULT_SEEDS = [0, 1, 2, 3, 4]

CURVE_COLOUR = "#0E7C7B"


def load_pooled(model, tag, seeds):
    """Pool over an explicit seed list rather than a `seed*` glob.

    A glob absorbs whatever is in results/, including files from a short
    diagnostic run or an earlier seed range, and reports the result as though it
    came from the intended sweep. Naming the seeds makes that impossible.
    """
    files = [f"{PROB_DIR}/probs_{tag}_{model}_seed{s}.npz" for s in seeds]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit(
            f"No prob files found for seeds {list(seeds)}: "
            f"{PROB_DIR}/probs_{tag}_{model}_seed<seed>.npz\n"
            f"Run run_ablation.py --task pneumonia first (it writes these).")
    if len(files) != len(seeds):
        print(f"  ! only {len(files)} of {len(seeds)} seed files present")
    stray = set(glob.glob(f"{PROB_DIR}/probs_{tag}_{model}_seed*.npz")) - set(files)
    for f in sorted(stray):
        print(f"  ! ignoring {os.path.basename(f)}: seed outside {list(seeds)}")
    ys, ps = [], []
    for f in files:
        d = np.load(f); ys.append(d["y"]); ps.append(d["prob"])
    return np.concatenate(ys).astype(int), np.concatenate(ps).astype(float), len(files)


def reliability_points(y, p, n_bins):
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)          # confidence in predicted class
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        xs.append(float(conf[m].mean()))
        ys.append(float(correct[m].mean()))
        ns.append(c)
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DSLiteDenseNet")
    ap.add_argument("--tag", default="pneumonia_lf1.0_scratch")
    ap.add_argument("--n_bins", "--bins", type=int, default=15,
                    help="Must match config/stats.py and the manuscript, which use 15.")
    ap.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS,
                    help="seeds to pool; must match the run_ablation.py call "
                         "that produced the probability files")
    args = ap.parse_args()

    y, p, nseed = load_pooled(args.model, args.tag, args.seeds)
    xs, ys, ns = reliability_points(y, p, args.n_bins)

    # Name the outputs after the run, not after one task: --tag selects the
    # predictions, so hard-coding "pneumonia" here silently overwrites a
    # pneumonia curve with a tuberculosis one.
    task = args.tag.split("_")[0]
    csv = f"reliability_curve_{args.model}_{task}.csv"
    with open(csv, "w") as f:
        f.write("bin_confidence_x,accuracy_y,n\n")
        for x, yy, n in zip(xs, ys, ns):
            f.write(f"{x:.4f},{yy:.4f},{int(n)}\n")

    print(f"Reliability curve ({args.model}, {nseed} seeds pooled, {args.n_bins} bins):")
    print("   x (Confidence)   y (Accuracy)")
    for x, yy in zip(xs, ys):
        print(f"   {x:.3f}            {yy:.3f}")
    print(f"\nSaved coordinates -> {csv}")

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=2.0, color="#9AA5B1")   # perfect calibration
    ax.plot(xs, ys, "-", lw=3.2, color=CURVE_COLOUR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])

    # Label and tick sizes are set for the printed size of this small panel.
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(axis='both', labelsize=11)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    pdf = f"reliability_curve_{args.model}_{task}.pdf"
    fig.savefig(pdf); plt.close(fig)
    print(f"Saved vector curve -> {pdf}")


if __name__ == "__main__":
    main()
