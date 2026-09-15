"""Score existing checkpoints on an external cohort, without retraining.

run_ablation.py writes one checkpoint per model and seed, tagged with the label
fraction. run_cross_source.py trains its own models and keeps none. So a
question of the form "how would these particular weights behave at another
hospital" can be answered from disk, provided the checkpoints are still there.

Two uses:

  A checkpoint trained at a reduced label fraction, scored externally. The
  internal label-efficiency curve says nothing about transfer, and this supplies
  the external half of the comparison:
      python eval_external.py --task pneumonia --tag pneumonia_lf0.05_scratch
      python eval_external.py --task pneumonia --tag pneumonia_lf1.0_scratch
  The two runs differ in one variable, so their difference is the quantity.

  An alternative positive label at the same site. The internal positives are the
  radiological descriptor "Lung Opacity" and CheXpert's default positives are the
  clinical label Pneumonia, so a cross-source comparison there crosses a
  definition as well as a site. CheXpert also carries a Lung Opacity column, and
  scoring the same weights against both separates the two:
      python eval_external.py --task pneumonia --source chexpert \\
             --pos_label "Lung Opacity" --chexpert_csv path/to/labels.csv \\
             --out_suffix _opacity
  Run the default label as well; a single label set on its own says nothing.

  ImageNet-initialised checkpoints from run_ablation.py --imagenet:
      python eval_external.py --task tb --tag tb_lf1.0_imagenet
  The negatives are the manifest's No Finding cases in both runs, while the
  positives are read from the label CSV, so the two differ in the definition of
  a positive and in nothing else. Prevalence differs between them, so compare
  AUROC rather than AP or ECE across the pair.

The threshold is taken from the internal validation split and carried unchanged,
as in the main experiments. Per-sample probabilities are written alongside the
metrics for operating_point.py.

Probability files are written as each pass finishes; the two CSVs only at the
end. If a run is interrupted, --from_npz rebuilds the CSVs from the probability
files already on disk without scoring anything again:

    python eval_external.py --task pneumonia --tag pneumonia_lf1.0_imagenet --from_npz

Absolute values will not reproduce the cross-source table of the manuscript,
which comes from its own training run. Compare within one invocation, or between
two that differ in a single variable.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from datasets import CXRDataset
from engine import predict, compute_metrics, youden_threshold
from models import build_model

MODELS = ["DenseNet121", "DenseNet121DS", "LiteDenseNet",
          "DSLiteDenseNet", "MobileNetV2", "ShuffleNetV2"]


def run_loader(model, dl, device, label, every=20):
    """predict() with a progress line, because these passes take minutes.

    An external cohort is tens of thousands of images and each one is decoded
    from DICOM or JPEG before it reaches the network, so a silent loop is
    indistinguishable from a hang. This prints often enough to tell them apart.
    """
    import numpy as _np
    model.eval()
    probs, labels = [], []
    n_batches = len(dl)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(dl, 1):
            x, y = batch[0], batch[1]
            p = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1]
            probs.append(p.cpu().numpy()); labels.append(_np.asarray(y))
            if i % every == 0 or i == n_batches:
                el = time.perf_counter() - t0
                eta = el / i * (n_batches - i)
                print(f"      {label}: batch {i}/{n_batches}  "
                      f"elapsed {el/60:.1f} min  eta {eta/60:.1f} min",
                      flush=True)
    return _np.concatenate(probs), _np.concatenate(labels)


def relabelled_chexpert(external, chexpert_csv, pos_label):
    """Swap CheXpert's positive class for another of its own label columns.

    Negatives stay the No Finding cases of the manifest, so the comparison is
    paired on the same negative set and only the definition of a positive
    changes.

    The positives are rebuilt from `chexpert_csv` rather than filtered out of
    the manifest. The manifest holds the pneumonia cohort, which is Pneumonia
    positives plus No Finding negatives; every case carrying the alternative
    label but neither of those two is absent from it. Selecting inside the
    manifest would therefore return Lung Opacity AND Pneumonia rather than
    Lung Opacity, which is a pruned pneumonia cohort and not a second label
    definition, and the difference between the two runs would not be the
    quantity the comparison is for.
    """
    lab = pd.read_csv(chexpert_csv)
    lab.columns = [str(c).strip() for c in lab.columns]
    if pos_label not in lab.columns:
        raise SystemExit(f"'{pos_label}' is not a column of {chexpert_csv}. "
                         f"Available: {[c for c in lab.columns][:25]}")
    key = "Path" if "Path" in lab.columns else lab.columns[0]

    # Same restriction to frontal projections that built the manifest, when the
    # column is present; a label file that lacks it is used as it stands.
    if "Frontal/Lateral" in lab.columns:
        lab = lab[lab["Frontal/Lateral"] == "Frontal"]
    pos = lab[pd.to_numeric(lab[pos_label], errors="coerce") == 1.0].copy()

    neg = external[(external.source == "chexpert") & (external.label == 0)].copy()
    neg_paths = set(neg.path.astype(str))

    # Resolve each relative CheXpert path the way data_manifests.build_chexpert
    # does, so the two cohorts address the same files on disk.
    root = C.RAW_DATA["chexpert"]["root"]

    def resolve(rel):
        rel = str(rel)
        return rel if rel.startswith(str(root)) else str(root.parent / rel)

    def pid(path):
        m = re.search(r"(patient\d+)", str(path))
        return m.group(1) if m else str(path)

    rows = []
    for rel in pos[key].astype(str):
        full = resolve(rel)
        rows.append(dict(image_id=rel, path=full, file_type="jpg", label=1,
                         patient_id=pid(rel), source="chexpert",
                         split="external"))
    pos_df = pd.DataFrame(rows, columns=["image_id", "path", "file_type",
                                         "label", "patient_id", "source",
                                         "split"])
    if pos_df.empty:
        raise SystemExit(
            f"No rows in {chexpert_csv} have '{pos_label}' == 1 "
            f"(after the frontal filter, if that column is present). "
            f"Check the column name and its encoding.")

    n_listed = len(pos_df)
    if n_listed:
        exists = pos_df.path.map(os.path.exists)
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"[relabel] {n_missing}/{n_listed} '{pos_label}' images are "
                  f"not on disk under {root}; they are dropped. "
                  f"e.g. {pos_df.loc[~exists, 'path'].head(2).tolist()}")
        pos_df = pos_df.loc[exists].reset_index(drop=True)

    # A No Finding row carrying the alternative label is counted once, as a
    # positive. CheXpert's No Finding is mutually exclusive with the pathology
    # columns, so this is normally empty.
    overlap = pos_df.path.isin(neg_paths).sum()
    if overlap:
        neg = neg[~neg.path.astype(str).isin(set(pos_df.path))]

    out = pd.concat([pos_df, neg], ignore_index=True)
    out["label"] = out["label"].astype(int)
    was_pos = int(external[(external.source == "chexpert")
                           & (external.label == 1)].label.sum())
    print(f"[relabel] '{pos_label}': {int(out.label.sum())} positive, "
          f"{int((out.label == 0).sum())} negative. The default cohort has "
          f"{was_pos} Pneumonia positives against the same negatives; the "
          f"positives here are read from {chexpert_csv}, not selected out of "
          f"the manifest, so the two runs differ in the label definition alone."
          + (f" {overlap} row(s) carry both labels and are counted as positive."
             if overlap else ""))
    if out.label.sum() == 0:
        raise SystemExit("No positives matched; check --chexpert_csv paths.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pneumonia", choices=["pneumonia", "tb"])
    ap.add_argument("--tag", default=None,
                    help="checkpoint tag, e.g. pneumonia_lf0.05_scratch "
                         "(default: <task>_lf1.0_scratch)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--source", nargs="+", default=None,
                    help="external sources to score (default: all in the manifest)")
    ap.add_argument("--pos_label", default=None,
                    help="alternative CheXpert positive column, e.g. 'Lung Opacity'")
    ap.add_argument("--chexpert_csv", default=None,
                    help="CheXpert label CSV, required with --pos_label")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4,
                    help="decoding, not the network, is the bottleneck here; "
                         "raise this to the number of physical cores")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first this-many images per source, "
                         "half of each class. For verifying the run before "
                         "committing to the full cohort: an even class split "
                         "is not the cohort prevalence, so AP, ECE and any "
                         "precision-based figure from a limited run are not "
                         "comparable with a full one. Sensitivity and "
                         "specificity do not depend on prevalence, but are "
                         "estimated here on a small head of each class")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--from_npz", action="store_true",
                    help="Rebuild the two CSVs from the per-seed .npz files "
                         "already in results/, scoring nothing. Each file holds "
                         "the labels, the probabilities and the threshold that "
                         "was used, so every column of the CSVs is recoverable "
                         "from it. Use after a run that was interrupted: the "
                         "probability files are written as each pass finishes, "
                         "but the CSVs only at the end, so a crash part-way "
                         "through leaves the completed work on disk and "
                         "unreadable. Combines files from several invocations.")
    args = ap.parse_args()

    tag = args.tag or f"{args.task}_lf1.0_scratch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  batch={args.batch}  workers={args.workers}", flush=True)
    if args.limit:
        print(f"[limit] the first {args.limit} images per source, half of each "
              f"class. The subsample prevalence is not the cohort prevalence, "
              f"so AP, ECE and any precision-based figure from this run are "
              f"not comparable with a full one. Sensitivity and specificity do "
              f"not depend on prevalence and AUROC is largely unaffected, but "
              f"all three are estimated on a fraction of the cohort.",
              flush=True)

    ext = pd.read_csv(C.MANIFEST_DIR / f"{args.task}_external.csv")
    internal = pd.read_csv(C.MANIFEST_DIR / f"{args.task}_internal.csv")
    val = internal[internal.split == "val"]

    if args.pos_label:
        if not args.chexpert_csv:
            raise SystemExit("--pos_label needs --chexpert_csv")
        ext = relabelled_chexpert(ext, args.chexpert_csv, args.pos_label)

    sources = args.source or sorted(ext.source.unique())
    rows = []

    if args.from_npz:
        for name in args.models:
            for seed in args.seeds:
                for src in sources:
                    f = (C.RESULTS_DIR /
                         f"extprobs_{tag}_{src}_{name}_seed{seed}{args.out_suffix}.npz")
                    if not f.exists():
                        print(f"  [skip] {f.name} not found")
                        continue
                    d = np.load(f)
                    y = np.asarray(d["y"]).astype(int)
                    p = np.asarray(d["prob"]).astype(float)
                    m = compute_metrics(y, p, float(np.asarray(d["thr"])))
                    m.update(model=name, seed=seed, eval_source=src, n=len(y),
                             prevalence=float(np.mean(y)), tag=tag)
                    rows.append(m)
                    print(f"  {name} seed{seed} {src}: AUROC={m['auroc']:.4f} "
                          f"AP={m['auprc']:.4f} sens={m['sens']:.3f} "
                          f"ECE={m['ece']:.4f}   [from {f.name}]")
        if not rows:
            raise SystemExit(
                f"No probability files matched tag '{tag}' in {C.RESULTS_DIR}. "
                f"--from_npz reads what an earlier run wrote; it cannot score "
                f"a model that has not been scored.")

    for name in ([] if args.from_npz else args.models):
        for seed in args.seeds:
            ckpt = C.CKPT_DIR / f"{name}_{tag}_seed{seed}.pth"
            if not ckpt.exists():
                print(f"  [skip] {ckpt.name} not found")
                continue
            model = build_model(name).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
            model.eval()

            # The threshold is chosen on the internal validation split, exactly
            # as in the main experiments, and carried unchanged to every site.
            print(f"  {name} seed{seed}: validation pass "
                  f"({len(val)} images) to fix the threshold", flush=True)
            vl = DataLoader(CXRDataset(val, train=False), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers)
            vp, vy = run_loader(model, vl, device, "validation")
            thr = youden_threshold(vy, vp)

            for src in sources:
                sub = ext[ext.source == src]
                if len(sub) == 0:
                    continue
                if args.limit:
                    sub = sub.groupby("label", group_keys=False).head(max(1, args.limit // 2))
                print(f"    {name} seed{seed} -> {src} ({len(sub)} images)", flush=True)
                dl = DataLoader(CXRDataset(sub, train=False), batch_size=args.batch,
                                shuffle=False, num_workers=args.workers)
                p, y = run_loader(model, dl, device, src)
                m = compute_metrics(y, p, thr)
                m.update(model=name, seed=seed, eval_source=src, n=len(y),
                         prevalence=float(np.mean(y)), tag=tag)
                rows.append(m)
                np.savez(C.RESULTS_DIR /
                         f"extprobs_{tag}_{src}_{name}_seed{seed}{args.out_suffix}.npz",
                         y=y, prob=p, thr=thr)
                print(f"  {name} seed{seed} {src}: AUROC={m['auroc']:.4f} "
                      f"AP={m['auprc']:.4f} sens={m['sens']:.3f} ECE={m['ece']:.4f}")
            # Release the weights and the cached blocks before the next model is
            # built. Six architectures are constructed in one process here, and
            # on a long external pass the allocator otherwise holds every one of
            # them until the process ends.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit(f"No checkpoints matched tag '{tag}' in {C.CKPT_DIR}.")
    df = pd.DataFrame(rows)
    df["limited"] = bool(args.limit)
    # The per-seed .npz names carry the source; these two CSVs did not, so a run
    # restricted with --source silently overwrote the CSVs of an earlier run on
    # a different site. A full run (no --source) keeps the plain name it always
    # had and holds every site in one file, distinguished by eval_source.
    # The same holds for a --models or --seeds subset: rescoring one model
    # would otherwise replace the CSVs holding all six.
    part = "" if not args.source else "_" + "-".join(sorted(args.source))
    if set(args.models) != set(MODELS):
        part += "_" + "-".join(args.models)
    if sorted(args.seeds) != sorted(C.SEEDS):
        part += "_seeds" + "-".join(str(x) for x in args.seeds)
    raw = C.RESULTS_DIR / f"external_{tag}{args.out_suffix}{part}_raw.csv"
    df.to_csv(raw, index=False)
    summary = (df.groupby(["model", "eval_source"])[["auroc", "auprc", "sens", "spec", "ece"]]
                 .agg(["mean", "std"]).round(4))
    print("\n" + summary.to_string())
    summary.to_csv(C.RESULTS_DIR / f"external_{tag}{args.out_suffix}{part}_summary.csv")
    print(f"\nWrote {raw} and the matching summary, plus per-seed probability files.")


if __name__ == "__main__":
    main()
