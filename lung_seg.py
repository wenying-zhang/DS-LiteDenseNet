"""Lung-field masks from the ChestX-Det PSPNet distributed by TorchXRayVision.

The mask defines the lung region for the Grad-CAM localisation metric and is
taken as the union of the left and right lung output channels. Montgomery ships
manual masks, which are preferred where available.

The mask is used for measurement only. Nothing in this repository crops an image
to the lung field: a crop estimated per image differs systematically between
institutions and would become a new site cue, which is the failure mode the
paper is about (Section 3.1 of the manuscript).

Segmenter: Lian et al., IEEE Trans. Med. Imaging 40(8):2042-2052, 2021.
Distribution: Cohen et al., MIDL 2022.

Needs torchxrayvision and a one-off download of about 260 MB of weights, so the
first run requires network access.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class LungSegmenter:
    def __init__(self, device=None):
        # Default to whatever is present rather than to CUDA. A hard "cuda"
        # default raises on a CPU-only machine, and both callers wrap the
        # constructor in a try/except, so the failure would surface as
        # "segmenter unavailable" and the lung metrics would silently vanish.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        import torchxrayvision as xrv
        self.xrv = xrv
        self.device = device
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        # find output channels whose target name mentions 'lung'
        targets = [t.lower() for t in self.model.targets]
        self.lung_idx = [i for i, t in enumerate(targets) if "lung" in t]
        if not self.lung_idx:
            raise RuntimeError(f"no lung channel found in PSPNet targets: {self.model.targets}")

    @torch.no_grad()
    def mask(self, pil_img, out_size=224, thr=0.5):
        """Return a binary lung mask (out_size x out_size, float32 in {0,1})."""
        g = np.array(pil_img.convert("L")).astype(np.float32)
        g = self.xrv.datasets.normalize(g, 255)          # -> roughly [-1024,1024]
        t = torch.from_numpy(g)[None, None]              # (1,1,H,W)
        t = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)
        t = t.to(self.device)
        logits = self.model(t)                           # (1, C, 512, 512)
        prob = torch.sigmoid(logits)[0, self.lung_idx].sum(0)  # union
        m = (prob > thr).float()[None, None]
        m = F.interpolate(m, size=(out_size, out_size), mode="nearest")
        return m[0, 0].cpu().numpy().astype(np.float32)


def montgomery_mask(image_id, left_dir, right_dir, out_size=224):
    """Use Montgomery's shipped manual masks where available (no model needed)."""
    lp = left_dir / f"{image_id}.png"
    rp = right_dir / f"{image_id}.png"
    if not (lp.exists() and rp.exists()):
        return None
    l = np.array(Image.open(lp).convert("L"))
    r = np.array(Image.open(rp).convert("L"))
    m = ((l > 127) | (r > 127)).astype(np.float32)
    m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
        (out_size, out_size), Image.NEAREST)) / 255.0
    return (m > 0.5).astype(np.float32)
