"""Build unified manifests from the raw corpora.

Writes to ./manifests/:
  pneumonia_internal.csv   RSNA; columns image_id, path, file_type, label,
                           patient_id, source, split
  pneumonia_external.csv   CheXpert and VinDr, split == 'external'
  tb_internal.csv          Shenzhen
  tb_external.csv          Montgomery and VinDr
  covid_internal.csv       COVIDGR-1.0, single source, no external arm

Properties the rest of the pipeline relies on:
  - label is binary, 0 for normal or no finding, 1 for pneumonia or TB;
  - patient_id is present on every row and the internal split is patient-wise
    and stratified, so no patient appears in more than one of train/val/test;
  - the source tag lets cross-source evaluation separate the hospitals.

Column names for each release are set in the constants below.

Run:  python data_manifests.py
"""
import os
import re
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

# --------- column names, as published in each release ------------------------
RSNA_CLASS_COL   = "class"        # in stage_2_detailed_class_info.csv
RSNA_PID_COL     = "patientId"
CHEXPERT_PNEU    = "Pneumonia"
CHEXPERT_NOFIND  = "No Finding"
CHEXPERT_FRONTAL = "Frontal/Lateral"   # value 'Frontal'
VINDR_ID_COL     = "image_id"
VINDR_RAD_COL    = "rad_id"
VINDR_PNEU       = "Pneumonia"
VINDR_TB         = "Tuberculosis"
VINDR_NOFIND     = "No finding"
# -----------------------------------------------------------------------------

TEST_FRAC = 0.15
VAL_FRAC  = 0.15   # of the whole; train = 0.70


def _patient_wise_split(df, seed=0):
    """Stratified, patient-wise split into train/val/test.
    We split the PATIENT table (one row per patient, label = patient majority),
    then map all of a patient's images to that patient's split -> group integrity.
    """
    pat = (df.groupby("patient_id")["label"]
             .agg(lambda s: int(round(s.mean())))   # majority label per patient
             .reset_index())
    strat = pat["label"] if pat["label"].nunique() > 1 else None
    pat_trainval, pat_test = train_test_split(
        pat, test_size=TEST_FRAC, random_state=seed, stratify=strat)
    strat2 = pat_trainval["label"] if pat_trainval["label"].nunique() > 1 else None
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    pat_train, pat_val = train_test_split(
        pat_trainval, test_size=val_rel, random_state=seed, stratify=strat2)
    split_of = {}
    for p in pat_train["patient_id"]: split_of[p] = "train"
    for p in pat_val["patient_id"]:   split_of[p] = "val"
    for p in pat_test["patient_id"]:  split_of[p] = "test"
    df = df.copy()
    df["split"] = df["patient_id"].map(split_of)
    return df


def _filter_existing(df, name):
    """Drop rows whose image file is missing from disk and report the count.

    Checking here rather than at load time keeps an unresolvable path from
    reaching the DataLoader, where it would surface only during training.
    """
    if df is None or len(df) == 0:
        return df
    exists = df["path"].map(lambda p: os.path.exists(p))
    n_missing = int((~exists).sum())
    if n_missing:
        missing_examples = df.loc[~exists, "path"].head(3).tolist()
        print(f"    [!] [{name}] {n_missing}/{len(df)} files NOT found on disk. "
              f"e.g. {missing_examples}")
    return df.loc[exists].reset_index(drop=True)


def _resolve_vindr_path(images_dir, img_id):
    """VinDr images may be shipped as .dicom, .dcm, .png, .jpg or .jpeg depending
    on how the release was exported. Probe the disk and return (path, file_type); fall
    back to .dicom (the canonical release) if none of them exist yet."""
    images_dir = Path(images_dir)
    for ext, ftype in ((".dicom", "dicom"), (".dcm", "dicom"),
                       (".png", "png"), (".jpg", "jpg"), (".jpeg", "jpg")):
        cand = images_dir / f"{img_id}{ext}"
        if cand.exists():
            return str(cand), ftype
    return str(images_dir / f"{img_id}.dicom"), "dicom"


# ------------------------------- RSNA ----------------------------------------
def build_rsna():
    p = C.RAW_DATA["rsna"]
    cls = pd.read_csv(p["class_csv"])
    # stage_2_detailed_class_info.csv carries one row per annotated bounding
    # box, so an image with two boxes appears twice with an identical class.
    # This collapses those duplicate rows; it does not merge several images of
    # one patient, because in RSNA the patientId IS the image identifier and
    # there is exactly one DICOM per patientId (see the path built below).
    cls = cls.drop_duplicates(subset=[RSNA_PID_COL])
    # 'Lung Opacity' = pneumonia (positive); 'Normal' = negative; drop the
    # ambiguous 'No Lung Opacity / Not Normal' middle class for a clean task.
    def lab(c):
        if c == "Lung Opacity": return 1
        if c == "Normal":       return 0
        return -1
    cls["label"] = cls[RSNA_CLASS_COL].map(lab)
    cls = cls[cls["label"] >= 0]
    rows = []
    for _, r in cls.iterrows():
        pid = r[RSNA_PID_COL]
        rows.append(dict(image_id=pid,
                         path=str(p["images"] / f"{pid}.dcm"),
                         file_type="dicom", label=int(r["label"]),
                         patient_id=pid, source="rsna"))
    df = pd.DataFrame(rows)
    print(f"[rsna] {len(df)} images  pos={df.label.sum()}  neg={(df.label==0).sum()}")
    return df


# ----------------------------- CheXpert --------------------------------------
def build_chexpert(task="pneumonia"):
    assert task == "pneumonia"
    p = C.RAW_DATA["chexpert"]
    df = pd.read_csv(p["train_csv"])
    if CHEXPERT_FRONTAL in df.columns:
        df = df[df[CHEXPERT_FRONTAL] == "Frontal"]
    pos = df[df[CHEXPERT_PNEU] == 1.0].copy();  pos["label"] = 1
    neg = df[df[CHEXPERT_NOFIND] == 1.0].copy(); neg["label"] = 0  # clean negatives
    out = pd.concat([pos, neg], ignore_index=True)
    def pid(path):
        m = re.search(r"(patient\d+)", str(path))
        return m.group(1) if m else str(path)
    rows = []
    for _, r in out.iterrows():
        rel = r["Path"]
        # CheXpert 'Path' starts with 'CheXpert-v1.0-small/...'; join to its parent
        full = (p["root"].parent / rel) if not str(rel).startswith(str(p["root"])) else rel
        rows.append(dict(image_id=str(rel), path=str(full), file_type="jpg",
                         label=int(r["label"]), patient_id=pid(rel), source="chexpert"))
    res = pd.DataFrame(rows); res["split"] = "external"
    print(f"[chexpert] {len(res)} ext images  pos={res.label.sum()}  neg={(res.label==0).sum()}")
    return res


# ------------------------------ VinDr ----------------------------------------
def _find_col(columns, *candidates):
    """Case-insensitive / whitespace-tolerant column lookup. Returns the real
    column name in `columns` matching any candidate, else None."""
    norm = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        hit = norm.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    return None


def build_vindr(task):
    """External test from VinDr-CXR image-level labels.

    VinDr-CXR image_labels_train.csv has one 0/1 column per pathology plus
    image_id and rad_id. In the TRAIN file each image is read by 3 radiologists
    (=> 3 rows/image); an already-aggregated / consensus file has 1 row/image.
    Labels are aggregated by strict majority, (n // 2) + 1 votes, which gives
    2 of 3 on the raw train file and 1 of 1 on a pre-aggregated one. A fixed
    threshold of two votes would silently discard every image of the latter.

    Column names are matched case-insensitively, and a missing column prints
    the header that was found rather than raising.
    """
    p = C.RAW_DATA["vindr"]
    raw = pd.read_csv(p["image_labels_csv"])
    raw.columns = [str(c).strip() for c in raw.columns]

    id_col   = _find_col(raw.columns, VINDR_ID_COL, "image_id", "imageid", "image id")
    pos_name = VINDR_PNEU if task == "pneumonia" else VINDR_TB
    pos_col  = _find_col(raw.columns, pos_name)
    nof_col  = _find_col(raw.columns, VINDR_NOFIND, "No finding", "No Finding", "NoFinding")

    if id_col is None or pos_col is None or nof_col is None:
        print(f"    [!] [vindr/{task}] could not locate required columns "
              f"(id={id_col}, positive '{pos_name}'->{pos_col}, nofinding={nof_col}).")
        print(f"        columns present: {list(raw.columns)}")
        return pd.DataFrame()      # _safe_build and the summary handle an empty frame

    for c in (pos_col, nof_col):   # guard against strings / NaN in the 0/1 columns
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    rows, n_pos, n_neg, n_skip = [], 0, 0, 0
    for img_id, sub in raw.groupby(id_col):
        n = len(sub)
        need = (n // 2) + 1                        # strict majority: 1-of-1 or 2-of-3
        if float(sub[pos_col].sum()) >= need:
            label = 1; n_pos += 1
        elif float(sub[nof_col].sum()) >= need:
            label = 0; n_neg += 1
        else:
            n_skip += 1
            continue                               # other disease / no consensus
        vpath, vftype = _resolve_vindr_path(p["images"], img_id)
        rows.append(dict(image_id=img_id, path=vpath, file_type=vftype,
                         label=label, patient_id=img_id, source="vindr"))

    res = pd.DataFrame(rows)
    if len(res):
        res["split"] = "external"
    reads = round(len(raw) / max(raw[id_col].nunique(), 1), 1)
    print(f"[vindr/{task}] {len(res)} ext images  pos={n_pos}  neg={n_neg}  "
          f"(skipped {n_skip} non-consensus; ~{reads} reads/image; pos_col='{pos_col}')")
    return res


# --------------------------- Shenzhen / Montgomery ---------------------------
def _tb_from_filenames(folder, source):
    import os
    rows = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = fn.rsplit(".", 1)[0]
        if stem.endswith("_1"):   label = 1
        elif stem.endswith("_0"): label = 0
        else:                     continue
        # Record the real container format. load_image dispatches on DICOM
        # versus everything else and PIL reads the header, so a mislabelled
        # JPEG still decoded correctly; but the manifest is published metadata
        # and should say what the file is. Both official NLM releases are PNG.
        ext = fn.rsplit(".", 1)[-1].lower()
        rows.append(dict(image_id=stem, path=str(folder / fn),
                         file_type="jpg" if ext == "jpeg" else ext,
                         label=label, patient_id=stem, source=source))
    return pd.DataFrame(rows)


def build_shenzhen():
    df = _tb_from_filenames(C.RAW_DATA["shenzhen"]["images"], "shenzhen")
    print(f"[shenzhen] {len(df)} images  pos={df.label.sum()}  neg={(df.label==0).sum()}")
    return df


def build_montgomery():
    df = _tb_from_filenames(C.RAW_DATA["montgomery"]["images"], "montgomery")
    df["split"] = "external"
    print(f"[montgomery] {len(df)} ext images  pos={df.label.sum()}  neg={(df.label==0).sum()}")
    return df


# --------------------------------- COVIDGR -----------------------------------
def build_covidgr():
    """COVID arm (OPTIONAL appendix). COVIDGR-1.0: positive and negative images
    come from the SAME hospital + SAME scanner, so a model cannot cheat on source
    -- the cleanest possible anti-shortcut design. Folder layout:
        <pos_dir>/*.jpg   (COVID+, label 1)
        <neg_dir>/*.jpg   (COVID-, label 0)
    One image per patient, so patient_id == image stem. There is NO external
    hospital arm for COVID by design: positives and negatives come from one
    source, so the task is a same-source check rather than a transfer test. The
    manifest is split patient-wise like the other internal sets and is consumed
    only by run_ablation.py, never by run_cross_source.py.
    """
    p = C.RAW_DATA["covidgr"]
    rows = []
    for label, key in ((1, "pos_dir"), (0, "neg_dir")):
        folder = Path(p[key])
        if not folder.exists():
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = fn.rsplit(".", 1)[0]
            ftype = "png" if fn.lower().endswith(".png") else "jpg"
            rows.append(dict(image_id=stem, path=str(folder / fn), file_type=ftype,
                             label=label, patient_id=stem, source="covidgr"))
    df = pd.DataFrame(rows)
    if len(df):
        print(f"[covidgr] {len(df)} images  covid+={df.label.sum()}  "
              f"non-covid={(df.label==0).sum()}")
    return df


# --------------------------------- main --------------------------------------
def _safe_build(fn, name, *args):
    """Run a builder, filter to files that exist, and never let one dataset's
    failure kill the others. Returns a possibly-empty DataFrame, so a missing
    external source degrades to '[SKIP]' instead of crashing the whole run."""
    try:
        df = fn(*args)
    except Exception as e:
        print(f"    [SKIP {name}] builder raised: {type(e).__name__}: {e}")
        return pd.DataFrame()
    return _filter_existing(df, name)


def main(seed=0):
    # ---- Pneumonia (MAIN): internal = RSNA; external = CheXpert + VinDr -------
    rsna = _safe_build(build_rsna, "rsna")
    if len(rsna) == 0:
        raise SystemExit(
            "[FATAL] RSNA produced 0 usable rows. Pneumonia is a main-line task, "
            "so this is a hard stop. Check RAW_DATA['rsna'] paths in config.py "
            "(and that the .dcm files actually exist there), then rerun.")
    rsna = _patient_wise_split(rsna, seed=seed)
    rsna.to_csv(C.MANIFEST_DIR / "pneumonia_internal.csv", index=False)
    ext_p_parts = [_safe_build(build_chexpert, "chexpert", "pneumonia")]
    if getattr(C, "USE_VINDR_EXTERNAL", False):
        ext_p_parts.append(_safe_build(build_vindr, "vindr/pneumonia", "pneumonia"))
    else:
        print("    [note] VinDr external is OFF (config.USE_VINDR_EXTERNAL=False); "
              "pneumonia external = CheXpert only.")
    ext_p = pd.concat(ext_p_parts, ignore_index=True)
    ext_p.to_csv(C.MANIFEST_DIR / "pneumonia_external.csv", index=False)

    # ---- TB (MAIN): internal = Shenzhen; external = Montgomery + VinDr --------
    shen = _safe_build(build_shenzhen, "shenzhen")
    if len(shen) == 0:
        raise SystemExit(
            "[FATAL] Shenzhen produced 0 usable rows. TB is a main-line task, so "
            "this is a hard stop. Check RAW_DATA['shenzhen'] path in config.py, "
            "then rerun.")
    shen = _patient_wise_split(shen, seed=seed)
    shen.to_csv(C.MANIFEST_DIR / "tb_internal.csv", index=False)
    ext_t_parts = [_safe_build(build_montgomery, "montgomery")]
    if getattr(C, "USE_VINDR_EXTERNAL", False):
        ext_t_parts.append(_safe_build(build_vindr, "vindr/tb", "tb"))
    else:
        print("    [note] VinDr external is OFF (config.USE_VINDR_EXTERNAL=False); "
              "tb external = Montgomery only.")
    ext_t = pd.concat(ext_t_parts, ignore_index=True)
    ext_t.to_csv(C.MANIFEST_DIR / "tb_external.csv", index=False)

    # ---- COVID (OPTIONAL appendix): internal = COVIDGR, no external arm -------
    covid = _safe_build(build_covidgr, "covidgr")
    if len(covid) == 0:
        print("    [SKIP covid] COVIDGR not found or empty; the COVID-19 arm "
              "is optional and the pneumonia and TB manifests are unaffected.")
    else:
        covid = _patient_wise_split(covid, seed=seed)
        covid.to_csv(C.MANIFEST_DIR / "covid_internal.csv", index=False)

    # ------------------------------ summary -----------------------------------
    print("\n" + "=" * 70)
    print("MANIFEST SUMMARY  --  verify row counts, that label has BOTH 0 and 1,")
    print("and that it is not wildly imbalanced. Any '[!] N/M files NOT found'")
    print("line above means a path is wrong; fix it before training.")
    print("=" * 70)
    internal = [("pneumonia_internal", ["split", "label"]),
                ("tb_internal",        ["split", "label"]),
                ("covid_internal",     ["split", "label"])]
    external = [("pneumonia_external", ["source", "label"]),
                ("tb_external",        ["source", "label"])]
    for name, by in internal + external:
        fp = C.MANIFEST_DIR / f"{name}.csv"
        if not fp.exists():
            print(f"\n[{name}]  (not written -- source skipped)")
            continue
        d = pd.read_csv(fp)
        print(f"\n[{name}]  {len(d)} rows")
        if len(d):
            print(d.groupby(by).size())


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=s)
