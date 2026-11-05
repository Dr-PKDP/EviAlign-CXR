"""Strictly-open configuration: label-derived explanation targets for NIH ChestX-ray14.
python tools/nih_templates.py --in manifests/nih.csv --out manifests/nih_templates.csv"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import pandas as pd

from evialign.findings import DEFAULT_CLASSES, DISPLAY

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
df = pd.read_csv(a.inp)


def rep(r):
    s = [f"There is {DISPLAY[c]}." for c in DEFAULT_CLASSES if r.get(f"y_{c}", 0) == 1]
    return " ".join(s) if s else "No acute cardiopulmonary abnormality."


df["report"] = df.apply(rep, axis=1)
df.to_csv(a.out, index=False)
print("wrote", a.out, len(df))
