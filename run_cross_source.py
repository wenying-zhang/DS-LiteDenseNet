"""Cross-source external validation.

Protocol:
  pneumonia: train on RSNA (USA)     -> test on RSNA internal,
             CheXpert (USA), VinDr (Vietnam)
  tb:        train on Shenzhen (China) -> test on Shenzhen internal,
             Montgomery (USA), VinDr (Vietnam)

Internal and external AUROC are reported side by side, so that the drop incurred
at each hospital is explicit. Two kinds of uncertainty are produced and they go
to different places: the per-seed bootstrap 95 per cent intervals, which carry
patient-level uncertainty, are printed and written to the raw CSV, while the
summary CSV and the LaTeX fragment for Table 5 carry the mean and standard
deviation across seeds, which is uncertainty over training runs. The manuscript
tabulates the second and quotes the first in the text.

Usage:
    python run_cross_source.py --task pneumonia
    python run_cross_source.py --task tb
"""
import os
import sys

# CUBLAS_WORKSPACE_CONFIG has to be in the environment before the CUDA context is
# created, which happens on the first torch call -- setting it later is silently
# ineffective. Doing it here, above `import torch`, means `--deterministic` works
# even when the shell variable was forgotten, and removes a step that is easy to
# get wrong (PowerShell's `set` is not cmd's `set`, and a new terminal tab does
# not inherit it).
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from datasets import CXRDataset, make_weighted_sampler, seed_worker
from models import build_model
from engine import set_seed, train_model, predict, compute_metrics, youden_threshold
from stats import bootstrap_auc_ci

_USE_VINDR = getattr(C, "USE_VINDR_EXTERNAL", False)
EXTERNAL = {
    "pneumonia": ["chexpert"] + (["vindr"] if _USE_VINDR else []),
    "tb":        ["montgomery"] + (["vindr"] if _USE_VINDR else []),
}
MODELS = ["DenseNet121", "MobileNetV2", "ShuffleNetV2", "DSLiteDenseNet"]


def loaders(task, seed):
    df = pd.read_csv(C.MANIFEST_DIR / f"{task}_internal.csv")
    tr = CXRDataset(df[df.split == "train"], train=True)
    va = CXRDataset(df[df.split == "val"], train=False)
    te = CXRDataset(df[df.split == "test"], train=False)
    ext = pd.read_csv(C.MANIFEST_DIR / f"{task}_external.csv")
    gen = torch.Generator().manual_seed(seed)
    s = make_weighted_sampler(tr.labels(), generator=gen)
    # Keep DataLoader workers alive across epochs; see run_ablation.get_loaders.
    _pw = C.NUM_WORKERS > 0
    L = lambda ds, **k: DataLoader(ds, batch_size=C.BATCH_SIZE,
                                   num_workers=C.NUM_WORKERS, pin_memory=True,
                                   persistent_workers=_pw,
                                   worker_init_fn=seed_worker, **k)
    # Same guard as run_ablation.get_loaders: a final training batch of exactly
    # one image makes nn.BatchNorm2d raise, so drop that batch, and only that
    # one. Neither internal training split used in the manuscript leaves a
    # remainder of one at batch size 32, so every reported run is unaffected;
    # the guard exists for a collaborator whose corpus differs in size.
    _drop_last = (len(tr) % C.BATCH_SIZE) == 1
    if _drop_last:
        print(f"  [loader] {len(tr)} training images leaves a final batch of "
              f"1 at batch size {C.BATCH_SIZE}; dropping it (BatchNorm needs "
              f"more than one sample per channel).")
    ext_lds = {}
    for src in EXTERNAL[task]:
        sub = ext[ext.source == src]
        if len(sub) == 0:
            print(f"   [skip] external source '{src}' has 0 rows -- not evaluating it.")
            continue
        ext_lds[src] = L(CXRDataset(sub, train=False), shuffle=False)
    return (L(tr, sampler=s, generator=gen, drop_last=_drop_last), L(va, shuffle=False),
            L(te, shuffle=False), ext_lds)


def main():
    ap = argparse.ArgumentParser()
    # COVIDGR is single-source and has no external hospital arm, so covid is not
    # offered here.
    ap.add_argument("--task", default="pneumonia", choices=["pneumonia", "tb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS,
                    help="seeds to train; the reported results use all five")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of models to run (default: all)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS,
                    help="override the training epoch budget")
    ap.add_argument("--deterministic", action="store_true",
                    help="make runs bit-reproducible on one machine; costs "
                         "roughly one third to one half again in training time")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_run = args.models if args.models else MODELS

    rows = []
    for model_name in models_to_run:
        for seed in args.seeds:
            set_seed(seed, args.deterministic)
            tr_ld, va_ld, te_ld, ext_lds = loaders(args.task, seed)
            model = build_model(model_name).to(device)
            print(f"\n[{model_name} | seed {seed}] training on internal source...")
            model, _ = train_model(model, tr_ld, va_ld, device, epochs=args.epochs)
            vprob, vy = predict(model, va_ld, device)
            thr = youden_threshold(vy, vprob)

            # internal test
            for name, ld in [("internal", te_ld)] + list(ext_lds.items()):
                prob, y = predict(model, ld, device)
                m = compute_metrics(y, prob, thr=thr)
                mean, lo, hi = bootstrap_auc_ci(y, prob, seed=seed)
                m.update(model=model_name, seed=seed, eval_source=name,
                         n=len(y), auroc_lo=lo, auroc_hi=hi)
                rows.append(m)
                print(f"   {name:10s} auroc={m['auroc']:.4f} "
                      f"[{lo:.3f},{hi:.3f}] sens={m['sens']:.3f} spec={m['spec']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / f"crosssource_{args.task}_raw.csv", index=False)

    pivot = (res.groupby(["model", "eval_source"])["auroc"]
                .agg(["mean", "std"]).reset_index())
    pivot.to_csv(C.RESULTS_DIR / f"crosssource_{args.task}_summary.csv", index=False)
    # Average precision is the informative metric at the low prevalence of the
    # external screening cohorts, so it is carried into the exported table too.
    # The summary CSV above holds AUROC only; ECE and AP live in the raw CSV.
    ap_pivot = (res.groupby(["model", "eval_source"])["auprc"]
                   .agg(["mean", "std"]).reset_index())

    order = ["internal"] + EXTERNAL[args.task]
    with open(C.RESULTS_DIR / f"crosssource_{args.task}.tex", "w") as f:
        f.write("\\begin{table}[ht]\\centering\\small\n")
        f.write(f"\\caption{{Cross-source AUROC on {args.task}: trained on the "
                f"internal source, evaluated on held-out hospitals "
                f"(mean$\\pm$std, {len(args.seeds)} seeds).}}\n")
        ap_src = EXTERNAL[args.task][-1]
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l " +
                "c " * (len(order) + 1) + "@{}}\n\\toprule\n")
        f.write("Model & " + " & ".join(order) +
                f" & AP ({ap_src})\\\\\n\\midrule\n")
        for m in models_to_run:
            cells = []
            for src in order:
                sub = pivot[(pivot.model == m) & (pivot.eval_source == src)]
                if len(sub):
                    cells.append(f"{sub['mean'].values[0]:.3f}$\\pm${sub['std'].values[0]:.3f}")
                else:
                    cells.append("--")
            sub = ap_pivot[(ap_pivot.model == m) & (ap_pivot.eval_source == ap_src)]
            cells.append(f"{sub['mean'].values[0]:.3f}$\\pm${sub['std'].values[0]:.3f}"
                         if len(sub) else "--")
            f.write(f"{m} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    print(f"\nWrote cross-source results to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
