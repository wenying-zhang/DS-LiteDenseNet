"""Paired comparisons of the saved per-seed AUROCs, one table per cohort.

The manuscript's paired panels --- Tables S13 and S14, and the rows collected in
Table S15 --- are all the same computation: for each baseline, take the five
paired per-seed AUROCs of that baseline and of DS-LiteDenseNet on one cohort,
form the difference, and report its mean, the two-sided paired t-test p-value and
the upper one-sided 95% confidence bound against the pre-specified margin
delta = 0.02. This script reads the CSVs the runs already wrote, so nothing is
re-trained, re-scored or re-read from the checkpoints.

Any `*_raw.csv` written by run_ablation.py, run_cross_source.py or
eval_external.py works. Rows are grouped by the `eval_source` column when it is
present, so one cross-source file yields its internal, CheXpert and VinDr cohorts
separately, and one external file yields the single cohort it scored.

Usage
-----
  # the cross-source paired panel behind Table 5 and Tables S3/S4
  python paired_external.py --files results/crosssource_pneumonia_raw.csv

  # the pneumonia ablation-checkpoint panel behind Table S14 (Section S17)
  python paired_external.py --files results/external_pneumonia_lf1.0_scratch_chexpert_raw.csv ^
      results/external_pneumonia_lf1.0_scratch_opacity_chexpert_raw.csv ^
      results/external_pneumonia_lf1.0_scratch_vindr_raw.csv

  # the tuberculosis panel behind Table S13 (Section S16). eval_external.py run
  # without --source writes both sites into one file, told apart by eval_source:
  python paired_external.py --files results/crosssource_tb_raw.csv ^
      results/external_tb_lf1.0_scratch_raw.csv
  # A per-site name such as external_tb_lf1.0_scratch_montgomery_raw.csv is also
  # accepted: when it does not exist, the combined file is read and filtered to
  # that site, and the substitution is printed.

  # everything at once, and write the rows out for Table S15
  python paired_external.py --files results/*_raw.csv --out results/paired_panels.csv
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import noninferiority_test, paired_ttest        # noqa: E402
from scipy import stats as _st                              # noqa: E402

_SITES = ("montgomery", "vindr", "chexpert", "shenzhen", "rsna")


def resolve(path):
    """Return (existing_path, site_filter) for a requested CSV.

    eval_external.py names its output after the --source list it was given, so a
    full run writes every site into external_<tag>_raw.csv while a restricted run
    writes external_<tag>_<site>_raw.csv. Asking for the per-site name when only
    the combined file exists used to drop the site silently; fall back to the
    combined file and keep only that site's rows.
    """
    if os.path.exists(path):
        return path, None
    d, name = os.path.split(path)
    m = re.match(r"^(external_.+)_(%s)_raw\.csv$" % "|".join(_SITES), name)
    if m:
        combined = os.path.join(d, m.group(1) + "_raw.csv")
        if os.path.exists(combined):
            return combined, m.group(2)
    return None, None


def load_panel(path):
    """Return {(cohort, model): {seed: auroc}} for one CSV."""
    df = pd.read_csv(path)
    if "auroc" not in df.columns or "model" not in df.columns:
        return {}
    if "eval_source" not in df.columns:
        df = df.assign(eval_source="internal")
    out = {}
    for _, r in df.iterrows():
        key = (str(r["eval_source"]), str(r["model"]))
        out.setdefault(key, {})[int(r["seed"])] = float(r["auroc"])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True,
                    help="one or more *_raw.csv files, or globs of them")
    ap.add_argument("--ref", default="DSLiteDenseNet",
                    help="the model every other row is compared against")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--tag", default=None,
                    help="keep only rows whose tag column equals this, when present")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f])
    rows = []
    seen = set()
    for requested in paths:
        path, site = resolve(requested)
        if path is None:
            print("missing:", requested)
            continue
        if site is not None:
            print("using %s, rows with eval_source == %s (for %s)"
                  % (path, site, os.path.basename(requested)))
        if (path, site) in seen:
            continue
        seen.add((path, site))
        df = pd.read_csv(path)
        if site is not None:
            if "eval_source" not in df.columns:
                print("  %s has no eval_source column; skipped" % path)
                continue
            df = df[df["eval_source"].astype(str) == site]
        if args.tag and "tag" in df.columns:
            df = df[df["tag"].astype(str) == args.tag]
        panels = {}
        for _, r in df.iterrows():
            key = (str(r.get("eval_source", "internal")), str(r["model"]))
            panels.setdefault(key, {})[int(r["seed"])] = float(r["auroc"])
        for (cohort, model), v in sorted(panels.items()):
            if model == args.ref:
                continue
            ref = panels.get((cohort, args.ref))
            if not ref:
                continue
            seeds = sorted(set(v) & set(ref))
            if len(seeds) < 2:
                continue
            base = [v[s] for s in seeds]
            ours = [ref[s] for s in seeds]
            mean_d, bound, _, ni = noninferiority_test(base, ours, margin=args.margin)
            _, p = paired_ttest(base, ours)
            tcrit = float(_st.t.ppf(0.95, len(seeds) - 1))
            near = bound == bound and abs(bound - args.margin) < 0.001
            rows.append(dict(
                file=os.path.basename(path), cohort=cohort, baseline=model,
                n_seeds=len(seeds),
                delta=round(mean_d, 4), p=round(p, 4),
                bound=round(bound, 4) if bound == bound else None,
                tcrit=round(tcrit, 3),
                ni_shown=int(bool(ni)),
                near_margin=int(bool(near)),
                per_seed=" ".join("%d:%.4f" % (s, v[s] - ref[s]) for s in seeds)))

    if not rows:
        raise SystemExit("no paired panel could be formed from those files")
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(out.to_string(index=False))
    print()
    # The literal percent sign must be doubled: this string goes through the
    # %-operator, and a bare "95% c" was read as a %c conversion (TypeError).
    print("delta = mean(base - ours) over the paired seeds; p = two-sided paired "
          "t-test; bound = upper one-sided 95%% confidence bound, mean + tcrit * se "
          "with tcrit = t(0.95, n_seeds - 1), against the margin delta = %.3f; "
          "ni_shown = 1 when the bound lies below it; near_margin = 1 when the "
          "bound is within 0.001 of it, so three decimals would round it onto the "
          "margin and the cell should be printed to four." % args.margin)
    if args.out:
        out.to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
