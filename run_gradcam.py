"""Figures 4 and 5, and Supplementary Figure S1: quantified Grad-CAM localisation.

Reports, per model and class:
  lung_energy   share of CAM mass inside the lung mask          (higher better)
  bg_energy     share of CAM mass in the outer 15 per cent border (lower better)
  box_energy    share inside the radiologist box, RSNA positives only
  pointing_hit  whether the CAM peak falls inside a box, RSNA positives only

Lung masks come from the ChestX-Det PSPNet, or from the Montgomery manual masks
where those exist. Qualitative overlay panels are saved alongside the table.

Usage:
    python run_gradcam.py --task pneumonia \
        --ckpt checkpoints/DSLiteDenseNet_pneumonia_lf1.0_scratch_seed0.pth

Checkpoints are named after the tag of the run that wrote them,
<model>_<task>_lf<fraction>_<scratch|ssl>_seed<seed>.pth, so the full-data
scratch weights are the ones named above.

A checkpoint with a different number of output classes can be evaluated by
passing --num_classes.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from PIL import Image

import config as C
from datasets import load_image, build_transforms
from models import build_model, get_target_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None; self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        w = self.grads.mean(dim=(2, 3), keepdim=True)         # GAP of grads
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def border_mask(size, frac=0.15):
    m = np.ones((size, size), np.float32); b = int(size * frac)
    m[b:-b, b:-b] = 0.0
    return m


_BOXES = None


def rsna_boxes(image_id):
    """Return list of [x,y,w,h] in original-pixel coords for an RSNA patientId.

    The annotation table is read once and indexed by patientId. Reading it per
    image turns a lookup into a full parse of a file with tens of thousands of
    rows, which dominates the runtime of the scoring loop.
    """
    global _BOXES
    if _BOXES is None:
        df = pd.read_csv(C.RAW_DATA["rsna"]["box_csv"])
        df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
        _BOXES = {k: v[["x", "y", "width", "height"]].values.tolist()
                  for k, v in df.groupby("patientId")}
    return _BOXES.get(image_id, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pneumonia", choices=["pneumonia", "tb"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="DSLiteDenseNet")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--pos_class", type=int, default=1, help="index of disease class")
    ap.add_argument("--n_per_class", type=int, default=120)
    ap.add_argument("--n_panels", type=int, default=2,
                    help="number of qualitative cases in the overlay panel")
    ap.add_argument("--cam_alpha", type=float, default=0.85,
                    help="overlay opacity at the CAM peak")
    ap.add_argument("--cam_floor", type=float, default=0.15,
                    help="overlay opacity where the CAM is zero")
    ap.add_argument("--out_suffix", default="",
                    help="Appended to every output name, so that repeating the "
                         "analysis across seeds does not overwrite the previous "
                         "seed. Use e.g. --out_suffix _seed1; the leading "
                         "underscore is part of the value.")
    ap.add_argument("--panel_px", type=int, default=512,
                    help="resolution of the DISPLAYED radiograph; the model "
                         "still sees C.IMG_SIZE")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.model, num_classes=args.num_classes).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd.get("model_state_dict", sd))
    cam = GradCAM(model, get_target_layer(model, args.model))
    tfm = build_transforms(train=False)

    seg = None
    try:
        from lung_seg import LungSegmenter
        seg = LungSegmenter(device=device)
    except Exception as e:
        print(f"[warn] PSPNet lung segmenter unavailable ({e}); "
              f"lung-energy will be skipped unless Montgomery masks exist.")

    df = pd.read_csv(C.MANIFEST_DIR / f"{args.task}_internal.csv")
    test = df[df.split == "test"]
    rows, panels = [], []
    for label in [0, 1]:
        sub = test[test.label == label].head(args.n_per_class)
        for _, r in sub.iterrows():
            img = load_image(r["path"], r["file_type"])
            x = tfm(img).unsqueeze(0).to(device)
            heat, cls = cam(x, class_idx=args.pos_class)

            # Hand-drawn masks take precedence over the segmenter wherever an
            # image has one. The reverse order leaves the manual branch
            # unreachable whenever the segmenter is installed.
            lung = None
            if r["source"] == "montgomery":
                from lung_seg import montgomery_mask
                lung = montgomery_mask(r["image_id"],
                                       C.RAW_DATA["montgomery"]["left_mask"],
                                       C.RAW_DATA["montgomery"]["right_mask"], C.IMG_SIZE)
            if lung is None and seg is not None:
                lung = seg.mask(img, out_size=C.IMG_SIZE)

            total = heat.sum() + 1e-8
            rec = dict(image_id=r["image_id"], source=r["source"], label=label)
            if lung is not None:
                rec["lung_energy"] = float((heat * lung).sum() / total)
                # Area fraction of this image's mask. A map with no spatial
                # information integrates to exactly this, so it is the
                # reference the energy share has to be read against; recording
                # it per image keeps the reference in the same file as the
                # measurement (see geometric_reference.py).
                rec["lung_area"] = float(np.asarray(lung, np.float32).mean())
            rec["bg_energy"] = float((heat * border_mask(C.IMG_SIZE)).sum() / total)

            if args.task == "pneumonia" and label == 1 and r["source"] == "rsna":
                boxes = rsna_boxes(r["image_id"])
                if boxes:
                    ow, oh = img.size  # PIL size = (W,H)
                    sx, sy = C.IMG_SIZE / ow, C.IMG_SIZE / oh
                    bmask = np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.float32)
                    for bx, by, bw, bh in boxes:
                        x0, y0 = int(bx * sx), int(by * sy)
                        x1, y1 = int((bx + bw) * sx), int((by + bh) * sy)
                        bmask[max(0, y0):y1, max(0, x0):x1] = 1.0
                    rec["box_energy"] = float((heat * bmask).sum() / total)
                    ay, ax = np.unravel_index(int(heat.argmax()), heat.shape)
                    rec["pointing_hit"] = int(bmask[ay, ax] > 0)
            rows.append(rec)

            if len(panels) < args.n_panels and label == 1:
                px = args.panel_px
                disp = np.array(img.convert("L").resize((px, px), Image.BICUBIC))
                heat_hi = np.asarray(Image.fromarray(
                    (heat * 255).astype(np.uint8)).resize((px, px), Image.BILINEAR)
                ).astype(np.float32) / 255.0
                lung_hi = None
                if lung is not None:
                    lung_hi = np.asarray(Image.fromarray(
                        (np.asarray(lung) * 255).astype(np.uint8)
                    ).resize((px, px), Image.NEAREST)).astype(np.float32) / 255.0
                panels.append((disp, heat_hi, lung_hi, r["image_id"]))

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / f"gradcam_{args.model}_{args.task}{args.out_suffix}_raw.csv", index=False)
    summ = res.groupby("label").mean(numeric_only=True)
    summ.to_csv(C.RESULTS_DIR / f"gradcam_{args.model}_{args.task}{args.out_suffix}_summary.csv")
    print("\nGrad-CAM localization summary (means):\n", summ)
    print("\nEach share is the fraction of activation mass inside a region and "
          "is interpretable only against that region's area; geometric_reference.py "
          "measures those areas. box_energy and pointing_hit are defined on the "
          "annotated positives only.")

    # Qualitative overlay panels.
    if panels:
        n = len(panels)
        # Column titles appear on the top row only, sized to survive scaling to
        # column width. Per-row image IDs go to a sidecar file, not the figure.
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), constrained_layout=True)
        if n == 1:
            axes = axes[None, :]
        col_titles = ["radiograph", "Grad-CAM", "overlay + lung field"]
        for i, (im, hm, lu, iid) in enumerate(panels):
            # Fixed vmin/vmax: otherwise imshow rescales the grey range per
            # panel, brightening the radiograph and washing out the overlay.
            axes[i, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[i, 1].imshow(hm, cmap="jet", vmin=0, vmax=1)
            axes[i, 2].imshow(im, cmap="gray", vmin=0, vmax=255)
            # Alpha tracks CAM intensity so cold regions stay transparent and
            # the anatomy remains visible underneath.
            a = np.clip(args.cam_floor
                        + (args.cam_alpha - args.cam_floor) * hm, 0.0, 1.0)
            try:
                axes[i, 2].imshow(hm, cmap="jet", alpha=a, vmin=0, vmax=1)
            except TypeError:                      # matplotlib < 3.4: scalar alpha
                axes[i, 2].imshow(hm, cmap="jet", alpha=float(args.cam_alpha),
                                  vmin=0, vmax=1)
            if lu is not None:
                axes[i, 2].contour(lu, levels=[0.5], colors="cyan", linewidths=1.6)
            if i == 0:
                for a, t in zip(axes[i], col_titles):
                    a.set_title(t, fontsize=17)
            for a in axes[i]:
                a.axis("off")
        stem = C.FIG_DIR / f"gradcam_{args.model}_{args.task}{args.out_suffix}_panels"
        fig.savefig(stem.with_suffix(".pdf"))              # vector text for print
        fig.savefig(stem.with_suffix(".png"), dpi=300)     # raster fallback
        plt.close(fig)
        with open(str(stem) + "_ids.txt", "w") as fh:
            fh.write("\n".join(f"row {i+1}: {p[3]}" for i, p in enumerate(panels)) + "\n")
        print(f"Saved qualitative panels -> {stem}.pdf (+ .png, + _ids.txt)")


if __name__ == "__main__":
    main()
