"""Two questions the saved probability files can answer without any retraining.

  A screening operating point. The manuscript fixes thresholds with Youden's J,
  which weights a missed case and a false alarm equally. Screening for a
  transmissible disease does not, and guidance for tuberculosis triage asks for
  high sensitivity rather than a balanced point. So: hold sensitivity at a
  target on the validation split, carry that threshold to the test set unchanged
  as the main experiments do, and report what specificity survives.

  Calibration under prior shift. Training uses a class-balanced sampler, so the
  network learns an approximately even prior, and the external cohorts sit near
  4% positive. Some of the external calibration error is therefore arithmetic
  rather than a failure of transfer. Shifting the logit by the log prior ratio
  removes exactly that component:

      logit_adj = logit_raw + log(pi_test / (1 - pi_test))
                            - log(pi_train / (1 - pi_train))

  Whatever calibration error survives the shift is the part transfer is
  responsible for. Reporting both separates the two.

Usage:
    # Section 4.5: threshold fixed on validation, carried to the locked test
    python operating_point.py --probs "results/probs_*.npz" \
        --val_probs "results/valprobs_*.npz" --tag pneumonia_lf1.0_scratch
    # the same for an external cohort; the validation files are the internal
    # ones of the run whose checkpoints eval_external.py scored
    python operating_point.py --probs "results/extprobs_*vindr*.npz" \
        --val_probs "results/valprobs_*.npz" --tag tb_lf1.0_scratch
    # prior-shift calibration only; needs no validation file
    python operating_point.py --probs "results/extprobs_*vindr*.npz" --train_prior 0.5

Files written by run_ablation.py hold internal test predictions; those written
by eval_external.py hold external ones and also carry the Youden threshold that
was used, which is checked against the validation file when both are present.
Reads only; writes a CSV when --out is given.

The sensitivity-constrained threshold is never chosen on the predictions it is
scored on unless --allow_test_threshold is passed. Without --val_probs the
sensitivity and specificity columns are left empty and only the calibration
columns are filled; with --val_probs, a scored file that has no validation
counterpart stops the run. Earlier versions of this script silently fell back to
selecting on the scored split.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from stats import expected_calibration_error


def threshold_at_sensitivity(y, prob, target):
    """Lowest-specificity-cost threshold reaching at least `target` sensitivity."""
    fpr, tpr, thr = roc_curve(y, prob)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return float(np.min(prob))
    return float(thr[ok[0]])


def at_threshold(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def prior_shift(prob, train_prior, test_prior, eps=1e-6):
    p = np.clip(prob, eps, 1 - eps)
    # A single-class cohort gives test_prior 0 or 1, and the log-odds of that is
    # infinite. Clip the priors on the same footing as the probabilities: the
    # result is meaningless either way, but it is a finite number and a printed
    # warning rather than a screen of divide-by-zero errors.
    tp = float(np.clip(test_prior, eps, 1 - eps))
    rp = float(np.clip(train_prior, eps, 1 - eps))
    logit = np.log(p / (1 - p))
    adj = (logit
           + np.log(tp / (1 - tp))
           - np.log(rp / (1 - rp)))
    return 1.0 / (1.0 + np.exp(-adj))


def youden_shortfall(y, prob, thr):
    """How much of Youden's J the threshold `thr` gives up on these predictions.

    Not a comparison of thresholds. Two executions of the same command can pick
    thresholds far apart and mean the same thing: J is often nearly flat near
    its maximum, so a change of 1e-5 in one probability moves the argmax across
    a plateau. Comparing the numbers then reports a difference where there is no
    disagreement -- which is what an earlier version of this check did, on three
    of thirty files, none of them actually stale.

    What a genuinely mismatched checkpoint does show is a threshold that is bad
    for these predictions. So evaluate the stored threshold on the validation
    predictions and return max(J) - J(thr): near zero whenever the two agree on
    the operating point, however far apart the numbers are.
    """
    fpr, tpr, thr_grid = roc_curve(y, prob)
    j_max = float(np.max(tpr - fpr))
    pred = (prob >= thr).astype(int)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return 0.0
    j_thr = float(pred[pos].mean() - pred[neg].mean())
    return j_max - j_thr


def parse_name(path):
    """Recover model, seed, run tag, suffix and evaluation source from a filename.

    Returns (model, seed, base_tag, suffix, source), or None.

    The tag matters. results/ accumulates probability files from every task and
    every label fraction, so grouping on model alone silently averages
    pneumonia, tuberculosis and COVID-19 together, and averages the label
    fractions on top of that. The resulting means are not a quantity.

    base_tag and suffix are returned separately because they answer different
    questions. base_tag (e.g. pneumonia_lf1.0_scratch) names the training run,
    and therefore the checkpoints and the validation predictions that belong to
    them. The suffix, from eval_external.py --out_suffix (e.g. _opacity), names
    a way of scoring those checkpoints. Grouping uses both, so two label
    definitions are never averaged; the validation lookup uses base_tag alone,
    because the validation split is the same whichever label the external
    cohort is scored against. Earlier versions concatenated the two
    before the lookup, so every suffixed file missed its validation counterpart
    and had its threshold chosen on the predictions it was scored on.
    """
    stem = Path(path).stem
    # Three prefixes are recognised: probs_ (internal test), extprobs_
    # (external test) and valprobs_ (validation, written for --val_probs).
    # Anchor the seed to the end of the name, allowing only eval_external.py's
    # --out_suffix after it. The character immediately after that underscore
    # has to be a letter, so a leftover such as ..._seed0_1.npz from an aborted
    # or exploratory run is still rejected rather than folded into seed 0,
    # where it would corrupt the mean. Underscores are allowed after that first
    # letter, so a suffix of several words such as _lung_opacity is read.
    # plot_reliability.py names its seeds for the same reason.
    m = re.search(r"^(?:ext|val)?probs_(.+?)_([A-Za-z0-9]+)_seed(\d+)"
                  r"(_[A-Za-z][A-Za-z0-9_]*)?$", stem)
    if not m:
        return None
    tag, model, seed = m.group(1), m.group(2), int(m.group(3))
    suffix = m.group(4) or ""
    src = "internal"
    for cand in ("chexpert", "vindr", "montgomery", "shenzhen", "rsna"):
        if tag.endswith("_" + cand):
            src = cand
            tag = tag[: -len("_" + cand)]
            break
    return model, seed, tag, suffix, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True,
                    help="glob over .npz files holding y and prob")
    ap.add_argument("--target_sens", type=float, default=0.90,
                    help="sensitivity to hold; 0.90 follows screening guidance")
    ap.add_argument("--train_prior", type=float, default=0.5,
                    help="positive rate the sampler presented during training")
    ap.add_argument("--test_prior", type=float, default=None,
                    help="positive rate at evaluation (default: measured per file)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds; leftovers are skipped by name")
    ap.add_argument("--tag", default=None,
                    help="restrict to one run tag, e.g. pneumonia_lf1.0_scratch")
    ap.add_argument("--val_probs", default=None,
                    help="Glob for the matching valprobs_*.npz written by "
                         "run_ablation.py. The sensitivity-constrained threshold "
                         "is fixed on these validation predictions and carried "
                         "unchanged to the file being scored. Required for the "
                         "sensitivity and specificity columns; a scored file with "
                         "no validation counterpart stops the run.")
    ap.add_argument("--allow_test_threshold", action="store_true",
                    help="Permit choosing the threshold on the predictions being "
                         "scored when no validation file matches. That is "
                         "test-set reuse, and the specificity it gives is an "
                         "optimistic ceiling rather than an estimate; such rows "
                         "read 'test (reused)' in thr_from. No result in the "
                         "manuscript uses it.")
    ap.add_argument("--j_tolerance", type=float, default=0.02,
                    help="How much of Youden's J the threshold stored in an "
                         "external file may give up on the validation "
                         "predictions before the file is reported as coming "
                         "from different weights (default 0.02)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.probs))
    if not files:
        raise SystemExit(f"No files matched {args.probs}")

    rows, skipped, unmatched, thr_mismatch = [], [], [], []
    # Index the validation predictions by (model, seed, tag) so the threshold
    # for each scored file comes from the same training run.
    val_lookup = {}
    if args.val_probs:
        for vf in sorted(glob.glob(args.val_probs)):
            vp = parse_name(vf)
            if vp is None:
                continue
            vmodel, vseed, vtag, vsuffix, _ = vp
            if not Path(vf).stem.startswith("valprobs_"):
                continue
            if vsuffix:
                print(f"  ! ignoring {Path(vf).name}: validation files carry no suffix")
                continue
            vd = np.load(vf)
            val_lookup[(vmodel, vseed, vtag)] = (
                np.asarray(vd["y"]).astype(int), np.asarray(vd["prob"]).astype(float))
        if not val_lookup:
            raise SystemExit(f"No usable files matched --val_probs {args.val_probs}")
        print(f"Thresholds fixed on {len(val_lookup)} validation files.\n")

    for f in files:
        if args.tag and args.tag not in Path(f).stem:
            continue
        d = np.load(f)
        y = np.asarray(d["y"]).astype(int)
        prob = np.asarray(d["prob"]).astype(float)
        if len(np.unique(y)) < 2:
            print(f"  [skip] {Path(f).name}: one class only")
            continue
        parsed = parse_name(f)
        if parsed is None:
            print(f"  ! ignoring {Path(f).name}: name does not end in _seed<N>")
            skipped.append(Path(f).name)
            continue
        model, seed, base_tag, suffix, src = parsed
        if Path(f).stem.startswith("valprobs_"):
            continue            # a validation file caught by a broad --probs glob
        tag = base_tag + suffix
        if args.seeds is not None and seed not in args.seeds:
            print(f"  ! ignoring {Path(f).name}: seed outside --seeds {args.seeds}")
            skipped.append(Path(f).name)
            continue
        pi = float(y.mean()) if args.test_prior is None else args.test_prior

        # Sensitivity-constrained point. The threshold is fixed on the
        # validation predictions of the same model, seed and training run, and
        # carried here unchanged, as Youden's J already is. The lookup uses
        # base_tag, not tag: an --out_suffix names a way of scoring the
        # checkpoints, not a different training run.
        vkey = (model, seed, base_tag)
        if vkey in val_lookup:
            vy, vprob = val_lookup[vkey]
            thr = threshold_at_sensitivity(vy, vprob, args.target_sens)
            thr_source = "validation"
            # eval_external.py stores the Youden threshold it fixed on the
            # validation split. Scoring it against the validation predictions
            # checks that the two came from the same checkpoint: a threshold
            # from different weights gives up much of the achievable J here.
            if "thr" in d.files:
                sf = youden_shortfall(vy, vprob, float(np.asarray(d["thr"])))
                if sf > args.j_tolerance:
                    thr_mismatch.append(f"{Path(f).name}  (gives up {sf:.3f} of J)")
            sens, spec = at_threshold(y, prob, thr)
        elif args.allow_test_threshold:
            thr = threshold_at_sensitivity(y, prob, args.target_sens)
            thr_source = "test (reused)"
            sens, spec = at_threshold(y, prob, thr)
        else:
            thr, sens, spec, thr_source = np.nan, np.nan, np.nan, "none"
            if val_lookup:
                unmatched.append(f"{Path(f).name}  (looked for valprobs_{base_tag}_{model}_seed{seed}.npz)")

        ece_raw = expected_calibration_error(y, prob)
        adj = prior_shift(prob, args.train_prior, pi)
        ece_adj = expected_calibration_error(y, adj)

        rows.append(dict(file=Path(f).name, model=model, seed=seed, tag=tag, source=src,
                         n=len(y), prevalence=pi,
                         thr_at_sens=thr, thr_from=thr_source, sens=sens, spec=spec,
                         ece_raw=ece_raw, ece_prior_adjusted=ece_adj,
                         ece_removed_by_prior=ece_raw - ece_adj))

    if unmatched:
        raise SystemExit(
            "No validation predictions for:\n  " + "\n  ".join(unmatched) +
            "\nThe threshold would otherwise be chosen on the predictions it is "
            "scored on. Point --val_probs at the files run_ablation.py wrote for "
            "these checkpoints, or pass --allow_test_threshold to accept a ceiling.")
    if thr_mismatch:
        print(f"\n  ! The threshold stored in these files gives up more than "
              f"{args.j_tolerance:.3f} of the Youden J available on the matching "
              f"validation predictions, so they were probably scored with "
              f"different weights from the ones that wrote those predictions. "
              f"Re-run eval_external.py on the current checkpoints:\n    "
              + "\n    ".join(thr_mismatch))
    df = pd.DataFrame(rows)
    if not df.empty and not val_lookup and not args.allow_test_threshold:
        print("\nNo --val_probs given: the sensitivity and specificity columns are "
              "left empty. Only the calibration columns are computed.")
    if df.empty:
        raise SystemExit(
            "No usable files. Every path matched by --probs was skipped: "
            "check the glob, and read the '! ignoring' lines above for "
            "names that do not end in _seed<N> or seeds outside --seeds.")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped by name; they are not in any mean below.")
    if df.duplicated(["tag", "source", "model", "seed"]).any():
        raise SystemExit("Two files map to the same run, source, model and seed. "
                         "Pass --seeds to name the intended set; averaging them "
                         "would double-count a single run.")
    pd.set_option("display.width", 220)
    print(f"\nSensitivity held at {args.target_sens:.2f}; "
          f"training prior assumed {args.train_prior:.2f}\n")
    print(df.drop(columns=["file"]).round(4).to_string(index=False))

    if len(df) > 1:
        # Group by tag as well as source, so runs from different tasks and
        # different label fractions are never averaged together.
        g = (df.groupby(["tag", "source", "model"])[["spec", "ece_raw", "ece_prior_adjusted"]]
               .agg(["mean", "std"]).round(4))
        print("\nMean over seeds, within each run:\n" + g.to_string())
        if df.tag.nunique() > 1:
            print(f"\n{df.tag.nunique()} distinct runs matched. Read each tag "
                  f"separately; a mean across tags is not a quantity.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    print("\nece_prior_adjusted is the calibration error remaining after "
          "correcting for the difference between the training and evaluation "
          "prevalence; the difference between the two columns is the component "
          "attributable to that prior shift alone.")


if __name__ == "__main__":
    main()
