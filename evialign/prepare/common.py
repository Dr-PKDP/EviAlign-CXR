"""Helpers shared by dataset adapters."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..findings import DEFAULT_CLASSES


def patient_split(df: pd.DataFrame, col: str = "patient_id", fracs=(0.8, 0.1, 0.1), seed: int = 0) -> pd.Series:
    """Deterministic patient-level split (no patient appears in two splits)."""
    pats = np.array(sorted(df[col].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    lab = {p: ("train" if i < a else "val" if i < b else "test") for i, p in enumerate(pats)}
    return df[col].astype(str).map(lab)


def write_manifest(df: pd.DataFrame, out: str, classes=DEFAULT_CLASSES):
    cols = ["image_path", "patient_id", "study_id", "dataset", "split", "report", "boxes", "anat_mask"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    ycols = [f"y_{c}" for c in classes if f"y_{c}" in df.columns]
    extra = [c for c in df.columns if c not in cols + ycols]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df[cols + ycols + extra].to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows; splits={df['split'].value_counts().to_dict()}")
    for c in ycols:
        v = df[c]
        print(f"  {c:24s} pos={int((v == 1).sum()):7d}  unc={int((v == -1).sum()):6d}  annotated={int(v.notna().sum())}")


def dicom_to_png(src: str, dst: str, max_side: int = 1024) -> tuple[int, int]:
    import pydicom
    from PIL import Image
    d = pydicom.dcmread(src)
    a = d.pixel_array.astype(np.float32)
    if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst)
    return d.pixel_array.shape[1], d.pixel_array.shape[0]
