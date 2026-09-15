"""Measure the geometric references that Table 6 energy shares are read against.

An energy share is only interpretable against what the same share would be for a
map carrying no spatial information. For a Grad-CAM map normalised to [0,1] and
spread uniformly over the frame, the fraction of its mass falling inside a region
equals that region's area fraction. The two references are therefore:

  border  fixed by construction. border_mask() zeroes a centre square after
          cutting int(size * 0.15) pixels from each edge, so at 224 px the
          centre is 158 px and the border covers 1 - (158/224)^2 = 0.502. The
          integer floor matters: the idealised 1 - 0.7^2 would give 0.510.

  lungs   not fixed, because it depends on the images and on the segmenter. It
          must be measured, and measured on the same images the Grad-CAM
          analysis scores -- the Shenzhen reference is not the RSNA one.

This script reproduces run_gradcam.py's image selection exactly (test split,
first --n_per_class of each label) and reports the mean mask area over it.

Usage:
    python geometric_reference.py                  # both tasks
    python geometric_reference.py --task tb

Prints only; writes nothing. Requires the image data and, for images without a
hand-drawn mask, the segmenter that run_gradcam.py uses.
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from datasets import load_image
from run_gradcam import border_mask


def reference_for(task, n_per_class):
    manifest = pd.read_csv(C.MANIFEST_DIR / f"{task}_internal.csv")
    test = manifest[manifest.split == "test"]

    try:
        from lung_seg import LungSegmenter
        seg = LungSegmenter()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  segmenter unavailable ({exc}); hand-drawn masks only")
        seg = None

    areas, area_label, skipped = [], [], 0
    for label in [0, 1]:
        sub = test[test.label == label].head(n_per_class)
        for _, r in sub.iterrows():
            img = load_image(r["path"], r["file_type"])
            mask = None
            if r["source"] == "montgomery":
                from lung_seg import montgomery_mask
                mask = montgomery_mask(r["image_id"],
                                       C.RAW_DATA["montgomery"]["left_mask"],
                                       C.RAW_DATA["montgomery"]["right_mask"],
                                       C.IMG_SIZE)
            if mask is None and seg is not None:
                mask = seg.mask(img, out_size=C.IMG_SIZE)
            if mask is None:
                skipped += 1
                continue
            areas.append(float(np.asarray(mask, np.float32).mean()))
            area_label.append(int(r["label"]))

    if not areas:
        raise SystemExit("No masks produced; cannot compute the reference.")
    a = np.asarray(areas)
    print(f"  lung reference : {a.mean():.4f}  "
          f"(sd {a.std(ddof=1):.4f}, n={len(a)}"
          + (f", {skipped} skipped)" if skipped else ")"))
    print(f"  rounded: ~{a.mean():.2f}")

    # A single pooled figure is the right reference for one share but the wrong
    # one for a difference between classes. If the segmented lung is larger on
    # positives, an uninformative map gains that much lung energy between the
    # classes, and a raw delta credits the model for it. Report both classes.
    lab = np.asarray(area_label)
    if lab.size == a.size and {0, 1} <= set(lab.tolist()):
        an, ap = a[lab == 0].mean(), a[lab == 1].mean()
        print(f"    by class     : (-)={an:.4f} (n={int((lab == 0).sum())})  "
              f"(+)={ap:.4f} (n={int((lab == 1).sum())})  delta={ap - an:+.4f}")
        print("    ^ this delta, not zero, is the null for the lung-energy "
              "delta in Table 6")

    # Box energy needs a reference of its own, on the same positives.
    if task != "pneumonia":
        return
    pos = test[test.label == 1].head(n_per_class)
    box_df = pd.read_csv(C.RAW_DATA["rsna"]["box_csv"])
    box_df = box_df[box_df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    by_id = {k: v[["x", "y", "width", "height"]].values.tolist()
             for k, v in box_df.groupby("patientId")}
    fracs = []
    for _, r in pos.iterrows():
        if r["source"] != "rsna":
            continue
        boxes = by_id.get(r["image_id"])
        if not boxes:
            continue
        img = load_image(r["path"], r["file_type"])
        ow, oh = img.size                       # PIL size = (W, H)
        sx, sy = C.IMG_SIZE / ow, C.IMG_SIZE / oh
        # Rasterise as run_gradcam.py does, so overlapping boxes are counted
        # once rather than summed.
        m = np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.float32)
        for bx, by, bw, bh in boxes:
            x0, y0 = int(bx * sx), int(by * sy)
            x1, y1 = int((bx + bw) * sx), int((by + bh) * sy)
            m[max(0, y0):y1, max(0, x0):x1] = 1.0
        fracs.append(float(m.mean()))
    if fracs:
        f = np.asarray(fracs)
        print(f"  box reference  : {f.mean():.4f}  "
              f"(sd {f.std(ddof=1):.4f}, n={len(f)} annotated positives)")
        print(f"  rounded: ~{f.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["pneumonia", "tb", "both"])
    ap.add_argument("--n_per_class", type=int, default=120,
                    help="must match the run_gradcam.py call being described")
    args = ap.parse_args()

    b = border_mask(C.IMG_SIZE).mean()
    print(f"border reference : {b:.4f}  "
          f"(fixed by construction at {C.IMG_SIZE} px; rounded ~{b:.2f})")

    for task in (["pneumonia", "tb"] if args.task == "both" else [args.task]):
        print(f"\n{task}:")
        reference_for(task, args.n_per_class)


if __name__ == "__main__":
    main()
