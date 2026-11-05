"""Prints reports where the rule labeler and CheXbert disagree, for tuning the finding patterns.
python tools/labeler_errors.py --manifest manifests/rexgradient.csv --finding cardiomegaly --n 25"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, re
import pandas as pd
from evialign.labeler import label_report

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--finding", required=True)
ap.add_argument("--n", type=int, default=25); ap.add_argument("--pool", type=int, default=20000)
a = ap.parse_args()
df = pd.read_csv(a.manifest, low_memory=False).drop_duplicates("study_id")
df = df[df["report"].fillna("").str.len() > 0].sample(n=min(a.pool, len(df)), random_state=1)
cb = (df[f"y_{a.finding}"].replace(-1, 1).fillna(0).values > 0).astype(int)
rl = pd.Series([label_report(t, [a.finding])[0] > 0 for t in df["report"]]).astype(int).values
def show(title, mask):
    sub = df[mask].head(a.n)
    print(f"\\n===== {title}: {int(mask.sum())} reports in pool of {len(df)} =====")
    for t in sub["report"]:
        print("-", re.sub(r"\\s+", " ", t)[:330])
show(f"{a.finding}: CheXbert positive, rule negative (rule misses)", (cb == 1) & (rl == 0))
show(f"{a.finding}: rule positive, CheXbert negative (rule over-calls)", (cb == 0) & (rl == 1))
