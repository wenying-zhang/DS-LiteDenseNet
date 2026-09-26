"""Does the outer border of the frame carry signal the classifiers use?

A Grad-CAM map says where the evidence appears to lie; it does not say whether the
classifier needs it there. This is the occlusion control of Section 4.4 of the
manuscript and Supplementary Section S21: it replaces one region of every image with
a constant and re-scores the same checkpoints on the same images, so the change in
AUROC is paired image by image.

  * If AUROC is unchanged, the region adds nothing the classifier uses given the
    rest of the image.
  * If AUROC falls, the region carries information the classifier uses. The fill is
    out of distribution, so the fall bounds the region's contribution from above
    rather than measuring it.

The regions below bound the effect from several sides.

  --region border   the outer 15% band, replaced by the mean of the retained
                    centre. This is the geometry the Grad-CAM border share is
                    defined on, so it tests the quantity the manuscript reports.
  --region centre   the 158-pixel centre block, replaced by the mean of the band
                    that surrounds it. Same area in the frame, opposite geometry,
                    and it gives the scale the border number has to be read on: a
                    region the model is known to need costs far more AUROC than a
                    region that turns out to be incidental.
  --region band_nonlung
                    the part of the outer band that lies OUTSIDE the lung mask,
                    filled like --region border. On a chest radiograph the band
                    also holds the lung apices and the lateral costophrenic
                    regions; this region removes the frame and its markers while
                    keeping that lung, so a loss here is a frame-level cue rather
                    than anatomy.
  --region band_lung
                    the complement: the part of the band INSIDE the lung mask.
                    border ~= band_nonlung + band_lung, which separates the two
                    readings of a band loss on tuberculosis.
  --region edge5    the outer 5% ring of the frame alone (11 px at 224).
  --region edge5_nonlung
                    that ring minus the lung: the collimation edge, the image
                    background, and any burned-in or lead marker that sits there.
  --region ring_nonlung
                    what lies between them, the 5% to 15% ring minus the lung,
                    which on a chest radiograph is mostly chest wall, shoulder and
                    neck soft tissue. Read against edge5_nonlung PER UNIT AREA,
                    this is what separates a cue at the frame edge (acquisition,
                    markers) from one spread over extrapulmonary tissue (body
                    habitus): the whole band cannot tell them apart.

The fill matters for the regions that touch the frame edge, which are largely
near-black background, so --fill is offered as well.

  --fill retained   the mean of everything kept (default). This is what
                    Tables S18-S21 of the Supplementary Material report.
  --fill region     the region's own mean. Its structure is destroyed and its
                    brightness preserved, so a loss that survives this fill is
                    not an artefact of laying a mid-grey patch over a dark
                    background. Report both for edge5* regions.

Every region prints the fraction of the frame it covers and the loss per unit
area. Compare regions of different size by the second, never by the first.

The two lung-aware regions need a lung mask per image, from the same segmenter
run_gradcam.py uses (Montgomery's hand masks where they exist). Masks are
computed once in the main process and cached under results/, so a second run is
fast; use --workers 0 on large external cohorts to avoid copying the cache into
every worker. Everything a worker needs is captured in the dataset object in the
parent process, so --workers > 0 is safe on Windows as well.

Only the input is altered; the checkpoints, the locked split, the threshold-free
metric and the preprocessing upstream of the region are exactly those of the main
text, so the two columns are paired image by image.

Geometry. The band is 15% of the frame at each edge: at 224 pixels that leaves a
158-pixel centre, as in Section 4.4. The fill is the mean of the region that is
kept, computed per channel after the transform, so the replacement carries the
intensity statistics the model expects at that point of the pipeline. Replacing a
region with a constant also introduces an edge the model never saw in training,
which is why the loss is read as an upper bound on the region's contribution
rather than as a measurement of it.

Usage
-----
  # the internal pneumonia ablation behind Table 3, locked test split
  python border_occlusion.py --task pneumonia

  # the two external pneumonia cohorts behind Table 5, same checkpoints
  python border_occlusion.py --task pneumonia --source chexpert vindr

  # the control condition: the same area, occluded at the centre instead
  python border_occlusion.py --task pneumonia --region centre

  # tuberculosis: separate the frame from the lung tissue inside the band
  python border_occlusion.py --task tb --region band_nonlung
  python border_occlusion.py --task tb --region band_lung

  # tuberculosis: split the non-lung band into the outer 5% ring and what lies
  # between it and the lung, with both fills, and the same on pneumonia as the
  # out-of-distribution control
  python border_occlusion.py --task tb --region edge5_nonlung --workers 0
  python border_occlusion.py --task tb --region edge5_nonlung --fill region --workers 0
  python border_occlusion.py --task tb --region ring_nonlung --workers 0
  python border_occlusion.py --task tb --region ring_nonlung --fill region --workers 0
  python border_occlusion.py --task pneumonia --region edge5_nonlung --fill region

  # tuberculosis, internal and the two external sites
  python border_occlusion.py --task tb
  python border_occlusion.py --task tb --source montgomery vindr

  # everything at once, and write the table out
  python border_occlusion.py --task pneumonia --source chexpert vindr ^
      --out results/border_occlusion_pneumonia.csv

The run is inference only: six models x five seeds x two passes over the split, so a
few minutes on the GPU that produced the paper.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from datasets import CXRDataset, seed_worker
from models import build_model


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
    n = len(p)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----------------------------------------------------------------- lung masks
REGIONS = ("border", "centre", "band_nonlung", "band_lung",
           "edge5", "edge5_nonlung", "ring_nonlung")
LUNG_REGIONS = ("band_nonlung", "band_lung", "edge5_nonlung", "ring_nonlung")
FILLS = ("retained", "region")


def _pack(m):
    return np.packbits(np.asarray(m, bool).ravel())


def _unpack(p, size=None):
    # size is passed in from the dataset, which captured it in the parent process:
    # a DataLoader worker on Windows re-imports this module, and config has been
    # seen to resolve to a different module there, so nothing in the worker path
    # may read C.IMG_SIZE.
    s = size or C.IMG_SIZE
    return np.unpackbits(p)[: s * s].reshape(s, s).astype(bool)


def lung_masks(df, cache_path):
    """{path: packed 224x224 lung mask} for every row, cached on disk."""
    from datasets import load_image
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = dict(z["masks"].item())
    todo = [r for _, r in df.iterrows() if r["path"] not in cache]
    if todo:
        from lung_seg import LungSegmenter, montgomery_mask
        seg = None
        print(f"segmenting {len(todo)} images (cache: {cache_path})")
        for k, r in enumerate(todo):
            img = load_image(r["path"], r["file_type"])
            m = None
            if r.get("source") == "montgomery":
                m = montgomery_mask(r["image_id"], C.RAW_DATA["montgomery"]["left_mask"],
                                    C.RAW_DATA["montgomery"]["right_mask"], C.IMG_SIZE)
            if m is None:
                seg = seg or LungSegmenter()
                m = seg.mask(img, out_size=C.IMG_SIZE)
            cache[r["path"]] = _pack(np.asarray(m) > 0.5)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(todo)}")
        np.savez_compressed(cache_path, masks=np.array(cache, dtype=object))
    b = int(C.IMG_SIZE * 0.15)
    band = np.ones((C.IMG_SIZE, C.IMG_SIZE), bool); band[b:-b, b:-b] = False
    frac = [(_unpack(cache[p], C.IMG_SIZE) & band).sum() / band.sum() for p in df["path"]]
    print(f"lung fills {np.mean(frac):.3f} of the outer band on average over "
          f"{len(frac)} images")
    return cache


# ----------------------------------------------------------------- the occlusion
def geometric_mask(region, b, b5, size=None):
    """The part of the frame a region covers, before any lung mask."""
    s = size or C.IMG_SIZE
    band = np.ones((s, s), bool); band[b:s - b, b:s - b] = False
    edge = np.ones((s, s), bool); edge[b5:s - b5, b5:s - b5] = False
    if region in ("border", "band_nonlung", "band_lung"):
        return band
    if region == "centre":
        return ~band
    if region in ("edge5", "edge5_nonlung"):
        return edge
    if region == "ring_nonlung":
        return band & ~edge
    raise SystemExit("unknown region " + region)


class OccludedRegion(CXRDataset):
    """Same dataset, with one region replaced by a constant.

    Geometry. ``border`` is the outer 15% band and ``centre`` the 158-pixel block
    it leaves; the two cover the same fraction of the frame. ``edge5`` is the
    outer 5% ring alone. The lung-aware regions intersect the geometry with the
    lung mask: ``band_nonlung`` and ``band_lung`` split the band, ``edge5_nonlung``
    is the outer ring outside the lung, and ``ring_nonlung`` is what lies between
    them --- the 5% to 15% ring outside the lung, which on a chest radiograph is
    mostly chest wall and shoulder soft tissue. Comparing the last two separates
    a cue at the frame edge (collimation, background, a burned-in or lead marker)
    from one spread over extrapulmonary tissue (body habitus), which the whole
    band cannot.

    Fill. ``--fill retained`` (the default, and what Tables S18--S21 report)
    replaces the region with the mean of everything kept, so the pixels that go in
    are typical of the rest of the image. ``--fill region`` replaces it with its
    own mean instead: the region's structure is destroyed while its brightness is
    preserved. The second matters for the outer ring, which is largely
    near-black background and collimation, where a mid-grey fill is itself far
    out of distribution; a loss that survives ``--fill region`` is not an artefact
    of the fill's brightness. Report both for any region that touches the frame
    edge.
    """

    def __init__(self, *args, border=0.15, region="border", masks=None,
                 edge=0.05, fill="retained", **kwargs):
        super().__init__(*args, **kwargs)
        if region not in REGIONS:
            raise SystemExit("--region must be one of " + ", ".join(REGIONS))
        if fill not in FILLS:
            raise SystemExit("--fill must be one of " + ", ".join(FILLS))
        if region in LUNG_REGIONS and masks is None:
            raise SystemExit("lung-aware regions need masks")
        self.region = region
        self.masks = masks
        self.fill = fill
        self.size = C.IMG_SIZE                     # captured here, not in a worker
        self.b = int(self.size * border)           # 224 * 0.15 -> 33, centre 158
        self.b5 = int(self.size * edge)
        if self.size - 2 * self.b != 158:
            raise SystemExit(
                "band geometry does not reproduce the 158-pixel centre used for the "
                "Grad-CAM border share; check --border and config.IMG_SIZE")
        self.geo = geometric_mask(region, self.b, self.b5, self.size)

    def mask_for(self, i):
        sel = self.geo
        if self.region in LUNG_REGIONS:
            lung = _unpack(self.masks[self.df.iloc[i]["path"]], self.size)
            sel = sel & (lung if self.region == "band_lung" else ~lung)
        return sel

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        x = x.clone()
        m = torch.from_numpy(self.mask_for(i))
        k = int(m.sum())
        if k == 0:
            return x, y
        # The fill is a constant per channel: the mean of everything kept, or the
        # mean of the region itself. Either way the region's structure is gone and
        # its edge is one the model never saw in training, so the change bounds
        # the region's contribution from above.
        src = ~m if self.fill == "retained" else m
        fill = x[:, src].mean(dim=1, keepdim=True)
        x[:, m] = fill.expand(-1, k)
        return x, y


# ----------------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = torch.softmax(model(x), dim=1)[:, 1]
        probs.append(p.float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(probs)


def loader_for(df, split, occlude, border, region, train=False, batch=32, workers=4,
               masks=None, edge=0.05, fill="retained"):
    cls = OccludedRegion if occlude else CXRDataset
    extra = ({"border": border, "region": region, "masks": masks,
              "edge": edge, "fill": fill} if occlude else {})
    ds = cls(df, split=split, train=train, **extra)
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                      worker_init_fn=seed_worker, generator=g)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="pneumonia", choices=["pneumonia", "tb"])
    ap.add_argument("--tag", default=None,
                    help="checkpoint tag (default <task>_lf1.0_scratch)")
    ap.add_argument("--models", nargs="+", default=C.ABLATION_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--source", nargs="+", default=None,
                    help="external sources from manifests/<task>_external.csv; "
                         "omit for the internal locked test split")
    ap.add_argument("--region", default="border", choices=list(REGIONS),
                    help="which region to replace: the outer band, the centre block, "
                         "the band outside / inside the lung mask, the outer 5%% ring, "
                         "that ring outside the lung, or the 5%%-15%% ring outside it")
    ap.add_argument("--edge", type=float, default=0.05,
                    help="width of the outer ring used by the edge5 regions, as a "
                         "fraction of the frame (default 0.05 -> 11 px at 224)")
    ap.add_argument("--fill", default="retained", choices=list(FILLS),
                    help="retained: fill with the mean of everything kept (default, "
                         "and what Tables S18-S21 report); region: fill with the "
                         "region's own mean, which preserves its brightness and so "
                         "separates a real cue from a fill artefact at the frame edge")
    ap.add_argument("--border", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()

    tag = args.tag or f"{args.task}_lf1.0_scratch"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"task={args.task}  tag={tag}  device={device}  region={args.region}  "
          f"border={args.border}  edge={args.edge}  fill={args.fill}")

    if args.source:
        ext = pd.read_csv(C.MANIFEST_DIR / f"{args.task}_external.csv")
        cohorts = [(src, ext[ext.source == src], None) for src in args.source]
        print("external cohorts: " + ", ".join(
            f"{src} (n={len(d)})" for src, d, _ in cohorts))
    else:
        man = pd.read_csv(C.MANIFEST_DIR / f"{args.task}_internal.csv")
        cohorts = [("internal test", man, "test")]
        print(f"internal locked test split (n={int((man.split == 'test').sum())})")
    print()

    masks = {}
    if args.region in LUNG_REGIONS:
        for cohort, df, split in cohorts:
            d = df if split is None else df[df.split == split]
            tagc = cohort.replace(" ", "_")
            masks.update(lung_masks(d, str(C.RESULTS_DIR / f"lungmask_{args.task}_{tagc}.npz")))

    # The replaced area, so that a loss can be read per unit area: the regions
    # below differ in size by more than an order of magnitude, and a larger loss
    # from a larger region is not by itself a stronger cue.
    b = int(C.IMG_SIZE * args.border); b5 = int(C.IMG_SIZE * args.edge)
    geo = geometric_mask(args.region, b, b5, C.IMG_SIZE)
    if args.region in LUNG_REGIONS:
        want_lung = args.region == "band_lung"
        fr = []
        for cohort, df, split in cohorts:
            d = df if split is None else df[df.split == split]
            for p in d["path"]:
                lung = _unpack(masks[p], C.IMG_SIZE)
                fr.append((geo & (lung if want_lung else ~lung)).sum() / geo.size)
        area = float(np.mean(fr))
    else:
        area = float(geo.sum() / geo.size)
    print(f"the replaced region covers {area:.4f} of the frame on average")
    print()

    rows = []
    for model_name in args.models:
        per_seed = {}
        for seed in args.seeds:
            ckpt = C.CKPT_DIR / f"{model_name}_{tag}_seed{seed}.pth"
            if not os.path.exists(ckpt):
                raise SystemExit(f"missing checkpoint {ckpt}")
            model = build_model(model_name)
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
            model.to(device)
            for cohort, df, split in cohorts:
                y0, p_plain = score(model, loader_for(df, split, False, args.border,
                                                      args.region, batch=args.batch,
                                                      workers=args.workers),
                                    device)
                y1, p_occ = score(model, loader_for(df, split, True, args.border,
                                                    args.region, batch=args.batch,
                                                    workers=args.workers,
                                                    masks=masks or None,
                                                    edge=args.edge, fill=args.fill),
                                  device)
                if not np.array_equal(y0, y1):
                    raise SystemExit("label order differs between the two passes")
                per_seed.setdefault(cohort, []).append(
                    (seed, auroc(y0, p_plain), auroc(y1, p_occ)))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for cohort, _df, _split in cohorts:
            v = per_seed[cohort]
            plain = np.array([a for _, a, _ in v])
            occ = np.array([b for _, _, b in v])
            rows.append(dict(
                task=args.task, cohort=cohort, model=model_name, region=args.region,
                n_seeds=len(v),
                auroc=float(plain.mean()), auroc_sd=float(plain.std(ddof=1)) if len(v) > 1 else 0.0,
                auroc_occluded=float(occ.mean()),
                auroc_occluded_sd=float(occ.std(ddof=1)) if len(v) > 1 else 0.0,
                delta=float((occ - plain).mean()),
                area=area,
                delta_per_area=float((occ - plain).mean() / area) if area > 0 else float("nan"),
                fill=args.fill, edge=args.edge,
                per_seed=" ".join("%.4f/%.4f" % (a, b) for _, a, b in v)))
            print("%-14s %-16s AUROC %.4f -> %.4f  (%+.4f, %+.3f per unit area)   %s" % (
                cohort, model_name, plain.mean(), occ.mean(), (occ - plain).mean(),
                (occ - plain).mean() / area if area > 0 else float("nan"),
                " ".join("%.4f/%.4f" % (a, b) for _, a, b in v)))

    print()
    print("AUROC is threshold-free and computed on the same images in both columns, "
          "so the pair is complete. A small delta means the replaced region carried "
          "little the classifier used; the fill is a constant, which introduces an "
          "edge the model never saw in training, so the delta bounds the region's "
          "contribution from above rather than measuring it. Run the other "
          "--region for the scale to read it on, and compare regions per unit "
          "area rather than by the raw delta. At the frame edge run --fill region "
          "as well: that fill keeps the region's own brightness, so a loss that "
          "survives it is not an artefact of a mid-grey patch over a near-black "
          "background.")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
