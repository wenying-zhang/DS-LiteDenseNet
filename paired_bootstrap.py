"""Patient-level paired bootstrap of the AUROC gap between DS-LiteDenseNet and each baseline.

Section 3.6 of the manuscript reports the one-sided 95% upper confidence bound on the
mean gap (baseline minus ours) from a paired t-test across the five training seeds, and
declares non-inferiority when that bound lies below the pre-specified margin. That bound
carries only the training-run sampling uncertainty: the test set is treated as fixed, so
the uncertainty that comes from having sampled 2230 test patients out of a population is
not in it. This script adds that component.

Three bounds are reported for every baseline, all on the same paired per-seed
predictions:

  bound_seed   one-sided 95% upper bound from the seed-level paired t-test (df = k-1).
               This is the number the manuscript currently quotes.
  bound_test   one-sided 95% upper bound from resampling patients within each seed with
               the seeds held fixed, i.e. test-set sampling uncertainty alone.
  bound_both   one-sided 95% upper bound from a two-level bootstrap that resamples both
               seeds and patients. This is the conservative combined bound, and the one
               to compare against the margin if the question is whether the verdict
               survives each source of variation being accounted for.

It also reports the fraction of two-level replicates whose gap reaches the margin,
P(gap >= margin), which is the bootstrap analogue of the power question the margin was
chosen to answer.

Patient grouping is read from the manifest, so the resampling unit is the patient rather
than the image. Where a corpus has one image per patient the two coincide, and the script
prints the grouping it found so that the reader can see which case applies.

Usage
-----
  # the internal pneumonia ablation behind Table 3 (the main non-inferiority claim)
  python paired_bootstrap.py --task pneumonia

  # the internal tuberculosis ablation behind Table 4
  python paired_bootstrap.py --task tb

  # the 5% label fraction, whose checkpoints feed the external evaluation
  python paired_bootstrap.py --task pneumonia --tag pneumonia_lf0.05_scratch

  # write the table to CSV as well
  python paired_bootstrap.py --task pneumonia --out results/noninferiority_bootstrap.csv

Nothing here retrains or re-scores anything: it reads the per-sample test probabilities
that run_ablation.py already writes (probs_<tag>_<model>_seed<n>.npz) and the manifests,
so it needs no GPU and takes seconds to a few minutes.
"""
import argparse
import collections
import csv
import math
import os

import numpy as np

import config as C


# --------------------------------------------------------------------------- AUROC
def auroc(y, p):
    """Rank-based AUROC with ties averaged, matching sklearn's roc_auc_score."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    n = len(p)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ------------------------------------------------------------------- small helpers
def load_npz(path):
    z = np.load(path)
    return z["y"], z["prob"]


def one_sided_t_bound(d, alpha=0.05, margin=None):
    """Upper (1-alpha) one-sided bound on mean(d) from a paired t across runs."""
    d = np.asarray(d, dtype=np.float64)
    k = len(d)
    if k < 2:
        return float("nan"), float("nan")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(k) if sd > 0 else 1e-12
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(1 - alpha, df=k - 1))
    except Exception:                                   # no scipy: normal approximation
        tcrit = 1.645 if alpha == 0.05 else 1.96
    return mean, mean + tcrit * se


def group_by_patient(patient_ids):
    """List of index arrays, one per patient, in first-appearance order."""
    groups = collections.OrderedDict()
    for i, pid in enumerate(patient_ids):
        groups.setdefault(pid, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def resample_indices(groups, rng):
    """Draw n_patients patients with replacement and concatenate their image indices."""
    n = len(groups)
    picked = rng.integers(0, n, n)
    return np.concatenate([groups[j] for j in picked])


# ------------------------------------------------------------------------ the test
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="pneumonia", choices=["pneumonia", "tb", "covid"],
                    help="which internal corpus and locked test split (default pneumonia)")
    ap.add_argument("--tag", default=None,
                    help="prefix of the probs_*.npz files (default <task>_lf1.0_scratch)")
    ap.add_argument("--ref", default="DSLiteDenseNet",
                    help="the compact model, minus which every gap is measured")
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="defaults to the other five ablation models")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="defaults to config.SEEDS")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="non-inferiority margin delta (default 0.02)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional CSV path for the table")
    args = ap.parse_args()

    tag = args.tag or f"{args.task}_lf1.0_scratch"
    seeds = args.seeds if args.seeds else list(C.SEEDS)
    baselines = args.baselines if args.baselines else [m for m in C.ABLATION_MODELS if m != args.ref]

    # ---- manifest: locked test split, in file order, which is the order the loader used
    man = C.MANIFEST_DIR / f"{args.task}_internal.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    te = [r for r in rows if r["split"] == "test"]
    y_man = np.array([int(r["label"]) for r in te], dtype=np.int64)
    patients = [r["patient_id"] for r in te]
    groups = group_by_patient(patients)
    per_pat = collections.Counter(len(g) for g in groups)
    print(f"task={args.task}  tag={tag}  test images={len(te)}  patients={len(groups)}"
          f"  images per patient={dict(sorted(per_pat.items()))}")
    if len(groups) == len(te):
        print("  note: one image per patient, so the patient-level bootstrap is the "
              "image-level bootstrap")
    print()

    # ---- read every seed's per-sample probabilities and check the labels line up
    data = {}
    for model in [args.ref] + baselines:
        for s in seeds:
            p = os.path.join(str(C.RESULTS_DIR), f"probs_{tag}_{model}_seed{s}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            y, prob = load_npz(p)
            if len(y) != len(te) or not np.array_equal(np.asarray(y).astype(np.int64), y_man):
                raise SystemExit(
                    f"labels in {os.path.basename(p)} do not match the manifest test split; "
                    "refusing to run")
            data[(model, s)] = np.asarray(prob, dtype=np.float64)

    # ---- per-seed observed AUROC, and the paired gaps
    auc = {}
    for model in [args.ref] + baselines:
        auc[model] = [auroc(y_man, data[(model, s)]) for s in seeds]
    gaps = {b: np.array([auc[b][i] - auc[args.ref][i] for i in range(len(seeds))])
            for b in baselines}

    print("per-seed AUROC")
    print("  %-16s %s" % (args.ref, " ".join("%.4f" % v for v in auc[args.ref])))
    for b in baselines:
        print("  %-16s %s" % (b, " ".join("%.4f" % v for v in auc[b])))
    print()

    # ---- bootstrap
    rng = np.random.default_rng(args.boot_seed)
    q = 100 * (1 - args.alpha)
    out_rows = []
    for b in baselines:
        ref_p = [data[(args.ref, s)] for s in seeds]
        base_p = [data[(b, s)] for s in seeds]

        # (a) test-set sampling uncertainty alone: seeds fixed, patients resampled
        boot_test = np.empty(args.n_boot)
        for it in range(args.n_boot):
            acc = 0.0
            for i in range(len(seeds)):
                idx = resample_indices(groups, rng)
                if y_man[idx].min() == y_man[idx].max():
                    acc += gaps[b][i]          # degenerate draw: keep the observed gap
                    continue
                acc += auroc(y_man[idx], base_p[i][idx]) - auroc(y_man[idx], ref_p[i][idx])
            boot_test[it] = acc / len(seeds)

        # (b) two-level: seeds resampled as well as patients
        boot_both = np.empty(args.n_boot)
        for it in range(args.n_boot):
            chosen = rng.integers(0, len(seeds), len(seeds))
            acc = 0.0
            for i in chosen:
                idx = resample_indices(groups, rng)
                if y_man[idx].min() == y_man[idx].max():
                    acc += gaps[b][i]
                    continue
                acc += auroc(y_man[idx], base_p[i][idx]) - auroc(y_man[idx], ref_p[i][idx])
            boot_both[it] = acc / len(seeds)

        mean_d, bound_seed = one_sided_t_bound(gaps[b], alpha=args.alpha)
        bound_test = float(np.percentile(boot_test, q))
        bound_both = float(np.percentile(boot_both, q))
        p_ge = float(np.mean(boot_both >= args.margin))

        out_rows.append(dict(baseline=b, delta=mean_d, sd=float(gaps[b].std(ddof=1)),
                             bound_seed=bound_seed, bound_test=bound_test,
                             bound_both=bound_both, p_ge_margin=p_ge,
                             ni_seed=int(bound_seed < args.margin),
                             ni_test=int(bound_test < args.margin),
                             ni_both=int(bound_both < args.margin)))

    hdr = ("%-16s %8s %8s %10s %10s %10s %8s" %
           ("baseline", "delta", "sd", "bound_seed", "bound_test", "bound_both", "P>=d"))
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        print("%-16s %+8.4f %8.4f %10.4f %10.4f %10.4f %8.4f" %
              (r["baseline"], r["delta"], r["sd"], r["bound_seed"],
               r["bound_test"], r["bound_both"], r["p_ge_margin"]))
    print()
    print("delta      = mean AUROC gap over the seeds (baseline minus ours)")
    print("bound_*    = one-sided %.0f%% upper bound on that mean, margin delta = %.3f"
          % (q, args.margin))
    print("P>=d       = fraction of two-level bootstrap replicates at or above the margin")
    print()
    print("non-inferiority verdicts: seed-level only       %s" %
          ["yes" if r["ni_seed"] else "no" for r in out_rows])
    print("                          adding test-set draws %s" %
          ["yes" if r["ni_test"] else "no" for r in out_rows])
    print("                          both sources          %s" %
          ["yes" if r["ni_both"] else "no" for r in out_rows])

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
