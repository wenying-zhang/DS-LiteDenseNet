"""Figure 6 and Supplementary Figure S2: reliability diagrams from the per-sample
probabilities saved by run_ablation.

Reads results/probs_<tag>_<model>_seed<seed>.npz and writes
figures/reliability_<tag>.png together with results/reliability_<tag>.csv, which
holds the ECE per model pooled over seeds. No GPU and no retraining are needed.

Convention follows Guo et al. (2017): confidence is the probability of the
predicted class, max(p, 1-p), and each bin compares mean confidence with
accuracy. The diagonal is perfect calibration; points below it indicate
overconfidence. The markers sit at each bin's mean confidence, which is the
value the accuracy is being compared with; the bars sit at the bin's geometric
centre, so that they tile the axis rather than overrunning it.

Usage:
    python plot_reliability.py --task pneumonia
    python plot_reliability.py --task tb
    python plot_reliability.py --tag pneumonia_lf0.1_scratch
"""
import argparse
import glob
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

import config as C
from stats import expected_calibration_error

DEFAULT_MODELS = ["DenseNet121", "DenseNet121DS", "LiteDenseNet",
                  "DSLiteDenseNet", "MobileNetV2", "ShuffleNetV2"]


def _load_pooled(tag, model, seeds):
    """Concatenate y/prob over an explicit seed list; None if none are present.

    The seed list is explicit rather than a `seed*` glob on purpose. A glob
    silently pools whatever happens to sit in results/, so a short diagnostic
    run, an aborted sweep or a file left over from an earlier seed range enters
    the estimate without any sign of it, and one model ends up pooled over a
    different number of runs than the others. Anything matching the pattern but
    outside `seeds` is reported and skipped.
    """
    files, missing = [], []
    for s in seeds:
        f = C.RESULTS_DIR / f"probs_{tag}_{model}_seed{s}.npz"
        (files if f.exists() else missing).append(f)
    if not files:
        return None
    stray = (set(glob.glob(str(C.RESULTS_DIR / f"probs_{tag}_{model}_seed*.npz")))
             - {str(f) for f in files})
    for f in sorted(stray):
        print(f"  ! ignoring {Path(f).name}: seed outside --seeds {list(seeds)}")
    for f in missing:
        print(f"  ! missing {f.name}")
    ys, ps = [], []
    for f in files:
        d = np.load(f)
        ys.append(d["y"]); ps.append(d["prob"])
    return np.concatenate(ys), np.concatenate(ps), len(files)


def _reliability_points(y, p, n_bins=15):
    """Per-bin mean confidence, accuracy, weight, and the bin's geometric centre.

    The mean confidence is where the marker belongs: it is the quantity the
    accuracy is compared against. The geometric centre is where the bar belongs.
    Drawing a bar of width 1/n_bins centred on the mean confidence puts its edge
    at mean + 1/(2*n_bins), which for a well-separated classifier exceeds 1.0 in
    the top bin and overlaps its neighbour lower down, so the bars stop
    partitioning the axis they are meant to partition.
    """
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ws, cs = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ws.append(m.mean())
        cs.append(0.5 * (lo + hi))
    return np.array(xs), np.array(ys), np.array(ws), np.array(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pneumonia",
                    choices=["pneumonia", "tb", "covid"])
    ap.add_argument("--tag", default=None,
                    help="Override the run tag; default '<task>_lf1.0_scratch'.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS,
                    help="seeds to pool; must match the run_ablation.py call "
                         "that produced the probability files")
    args = ap.parse_args()
    tag = args.tag or f"{args.task}_lf1.0_scratch"

    present = []
    for m in args.models:
        got = _load_pooled(tag, m, args.seeds)
        if got is not None:
            present.append((m, got[0], got[1], got[2]))
    # Every model must be pooled over the same seeds, or the ECE column compares
    # estimates built from different amounts of data.
    counts = {m: n for m, _, _, n in present}
    if len(set(counts.values())) > 1:
        raise SystemExit(
            "Models are pooled over unequal numbers of seed files, so their ECE "
            f"values are not comparable: {counts}. Re-run the ablation for the "
            "models with missing seeds, or pass --seeds to name the intended set.")
    if not present:
        raise SystemExit(
            f"No prob files found for tag '{tag}'. Run run_ablation.py first "
            f"(it writes results/probs_{tag}_<model>_seed<seed>.npz).")

    # Font sizes target the printed size: a 12 inch canvas placed at roughly
    # 0.88 column width, so scaled by about one half.
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.titlesize": 16})
    ncol = min(3, len(present))
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow),
                             squeeze=False, constrained_layout=True)
    rows = []
    for k, (m, y, p, nseed) in enumerate(present):
        ax = axes[k // ncol][k % ncol]
        xs, ys, ws, cs = _reliability_points(y, p, args.n_bins)
        ece = expected_calibration_error(y, p, args.n_bins)
        rows.append(dict(model=m, ece=ece, n=len(y), seeds=nseed))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.bar(cs, ys, width=1.0 / args.n_bins, alpha=0.35, edgecolor="k",
               align="center", label="accuracy")
        ax.plot(xs, ys, "o-", ms=4)
        ax.set_title(f"{m}\nECE={ece:.3f} ({nseed} seeds pooled)", fontsize=16)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    for k in range(len(present), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    out = C.FIG_DIR / f"reliability_{tag}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / f"reliability_{tag}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved figure -> {out}")
    print(f"Saved ECE table -> {C.RESULTS_DIR / f'reliability_{tag}.csv'}")


if __name__ == "__main__":
    main()
