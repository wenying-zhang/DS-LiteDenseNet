"""Paths, hyper-parameters and preprocessing switches shared by every script.

Only the dataset paths normally need changing: point DATA_ROOT at the directory
holding the raw corpora, or edit individual entries in RAW_DATA if a dataset
sits elsewhere.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Raw data is expected under DATA_ROOT, by default a "data" directory next to
# this file. Defining it from the location of this file, rather than as a
# hard-coded drive or home-directory path, lets the tree work unchanged on
# Windows and POSIX. The resolved value is absolute on whichever machine runs
# it, so the manifests data_manifests.py writes hold that machine's absolute
# paths; they are local artefacts, excluded by .gitignore, and every clone
# rebuilds its own with `python data_manifests.py`.
DATA_ROOT = PROJECT_ROOT / "data"

MANIFEST_DIR = PROJECT_ROOT / "manifests"
RESULTS_DIR  = PROJECT_ROOT / "results"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
FIG_DIR      = PROJECT_ROOT / "figures"
for _d in (MANIFEST_DIR, RESULTS_DIR, CKPT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Sub-paths are relative to DATA_ROOT. Column names live in data_manifests.py.
RAW_DATA = {
    # RSNA Pneumonia Detection Challenge 2018. DICOM.
    "rsna": {
        "images":    DATA_ROOT / "rsna" / "stage_2_train_images",
        "class_csv": DATA_ROOT / "rsna" / "stage_2_detailed_class_info.csv",
        "box_csv":   DATA_ROOT / "rsna" / "stage_2_train_labels.csv",
    },
    # CheXpert v1.0 small. JPG.
    "chexpert": {
        "root":      DATA_ROOT / "chexpert" / "CheXpert-v1.0-small",
        "train_csv": DATA_ROOT / "chexpert" / "CheXpert-v1.0-small" / "train.csv",
        "valid_csv": DATA_ROOT / "chexpert" / "CheXpert-v1.0-small" / "valid.csv",
    },
    # VinDr-CXR v1.0.0. DICOM. External test only.
    "vindr": {
        "images":           DATA_ROOT / "vindr" / "train",
        "image_labels_csv": DATA_ROOT / "vindr" / "image_labels_train.csv",
    },
    # Shenzhen TB set. PNG, label encoded in the filename suffix (_0 / _1).
    "shenzhen": {
        "images": DATA_ROOT / "tb" / "ChinaSet_AllFiles" / "CXR_png",
    },
    # Montgomery County TB set. PNG, ships with manual lung masks.
    "montgomery": {
        "images":    DATA_ROOT / "tb" / "MontgomerySet" / "CXR_png",
        "left_mask": DATA_ROOT / "tb" / "MontgomerySet" / "ManualMask" / "leftMask",
        "right_mask":DATA_ROOT / "tb" / "MontgomerySet" / "ManualMask" / "rightMask",
    },
    # COVIDGR-1.0, single-source COVID arm. P/ positive, N/ negative. Both
    # partitions come from one hospital and one scanner, so the classes cannot
    # be separated by acquisition source.
    "covidgr": {
        "pos_dir": DATA_ROOT / "COVIDGR" / "P",
        "neg_dir": DATA_ROOT / "COVIDGR" / "N",
    },
    # NIH ChestX-ray14, optional. RSNA is a re-labelled subset of it, so the two
    # must never be paired across a train/test boundary.
    "nih": {
        "images":    DATA_ROOT / "nih" / "images",
        "label_csv": DATA_ROOT / "nih" / "Data_Entry_2017.csv",
    },
}

# External validation sources.
#
# All reported VinDr results were produced with this flag set. It requires the
# adult VinDr-CXR v1.0.0 release (15,000 training images, PhysioNet credentialed
# access, image-level Pneumonia / Tuberculosis / No finding labels).
#
# The more easily obtained mirror is VinDr-PCXR, a paediatric set of 7,728
# images from children under 10. Substituting it would confound a change of
# hospital with a change of age population. Set the flag to False in that case;
# external validation then runs on CheXpert (pneumonia) and Montgomery (TB).
USE_VINDR_EXTERNAL = True

# Preprocessing. CLAHE followed by per-image standardisation brings corpora
# acquired on different equipment into a common input distribution; Shenzhen and
# Montgomery differ by roughly 0.8 normalised intensity units without it.
# Changing either flag invalidates previously computed results.
USE_CLAHE  = True
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
# Per-image standardisation replaces the fixed ImageNet normalisation, which
# carries no benefit here because the models train from scratch.
PER_IMAGE_NORM = True

# Training.
IMG_SIZE       = 224
SEEDS          = [0, 1, 2, 3, 4]
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 5e-4
WEIGHT_DECAY   = 1e-4
EARLY_PATIENCE = 15                 # epochs without val-AUROC improvement
NUM_WORKERS    = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Four ablation variants plus two lightweight baselines.
ABLATION_MODELS = [
    "DenseNet121",     # [6,12,24,16], standard convolution
    "DenseNet121DS",   # [6,12,24,16], depthwise separable
    "LiteDenseNet",    # [4,6,8,6],    standard convolution
    "DSLiteDenseNet",  # [4,6,8,6],    depthwise separable
    "MobileNetV2",
    "ShuffleNetV2",
]
