"""Dataset class and image transforms, driven by the unified manifests."""
import random
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config as C


def read_dicom_image(path):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception as exc:                                    # noqa: BLE001
        # Not fatal: the raw pixel array is still usable and the windowing
        # below handles its range. Warn rather than pass silently, so a file
        # whose LUT never applies is visible instead of quietly different.
        warnings.warn(f"VOI LUT not applied to {path}: {exc}", RuntimeWarning)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr).convert("L")


def _pil_to_gray_uint8(pil):
    """Convert a PIL image of any bit depth to 8-bit grayscale.

    8-bit inputs go through convert('L') unchanged. Higher bit depths (modes I,
    I;16*, F, as found in part of the Montgomery set) are windowed and rescaled
    per image instead, since convert('L') clips them towards white and leaves
    an almost blank frame.
    """
    if pil.mode not in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        return pil.convert("L")
    arr = np.asarray(pil).astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])      # robust window; ignore hot pixels
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return Image.fromarray(np.zeros(arr.shape, np.uint8))
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _apply_clahe(pil_l):
    """Contrast-limited adaptive histogram equalisation on an 8-bit image.

    Equalising local contrast puts corpora acquired on different equipment into
    a comparable input distribution; Shenzhen and Montgomery differ by about
    0.8 normalised intensity units before this step.
    """
    import cv2
    # One thread per worker process. OpenCV defaults to using every core, and
    # with several DataLoader workers each doing the same the processes
    # oversubscribe the machine and throughput collapses.
    cv2.setNumThreads(0)
    arr = np.asarray(pil_l, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=getattr(C, "CLAHE_CLIP", 2.0),
                            tileGridSize=(getattr(C, "CLAHE_TILE", 8),
                                          getattr(C, "CLAHE_TILE", 8)))
    return Image.fromarray(clahe.apply(arr))


def load_image(path, file_type):
    if str(file_type).lower() == "dicom":
        img = read_dicom_image(path)
    else:
        img = _pil_to_gray_uint8(Image.open(path))
    if getattr(C, "USE_CLAHE", False):
        img = _apply_clahe(img)
    return img.convert("RGB")


class _PerImageStandardize:
    """Standardise each image tensor to zero mean and unit variance.

    This removes per-image, and hence per-corpus, exposure and brightness as a
    confound. The models train from scratch, so the fixed ImageNet statistics
    would contribute nothing beyond an affine transform that does not equalise
    across sources. Defined at module level so it stays picklable under
    DataLoader workers.
    """
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)


def _normalizer():
    if getattr(C, "PER_IMAGE_NORM", False):
        return _PerImageStandardize()
    return transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)


def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            _normalizer(),
        ])
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        _normalizer(),
    ])


class CXRDataset(Dataset):
    def __init__(self, manifest, split=None, source=None, train=False,
                 transform=None, return_meta=False):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if split is not None:
            df = df[df["split"] == split]
        if source is not None:
            df = df[df["source"] == source]
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform is not None else build_transforms(train)
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def labels(self):
        return self.df["label"].to_numpy().astype(int)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_image(r["path"], r["file_type"])
        x = self.transform(img)
        y = int(r["label"])
        if self.return_meta:
            return x, y, {"image_id": r["image_id"], "source": r["source"],
                          "path": r["path"], "file_type": r["file_type"]}
        return x, y


def make_weighted_sampler(labels, generator=None):
    """Class-balanced sampling; the RSNA pneumonia corpus is 40 per cent positive.

    Pass a seeded generator to make the draw order reproducible.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    w_per_class = 1.0 / np.clip(counts, 1, None)
    weights = w_per_class[labels]
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels),
        replacement=True, generator=generator)


def seed_worker(worker_id):
    """Give each DataLoader worker a derived, reproducible seed.

    Workers are separate processes and do not inherit the numpy or random state
    of the parent, so without this the augmentation stream differs between runs.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
