"""Agreement between the rule labeler and CheXbert labels on the training reports (QC for the token masks)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from evialign.labeler import label_report

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()
df = pd.read_csv(a.manifest, low_memory=False).drop_duplicates("study_id")
df = df[df["report"].fillna("").str.len() > 0].sample(n=min(a.n, len(df)), random_state=0)
cls = ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax", "nodule_mass", "lung_opacity"]
rule = pd.DataFrame([label_report(t, cls) for t in df["report"]], columns=cls)
rows = []
for c in cls:
    cb = (df[f"y_{c}"].replace(-1, 1).fillna(0).values > 0).astype(int)
    rl = (rule[c].values > 0).astype(int)
    rows.append({"finding": c, "chexbert_pos": int(cb.sum()), "rule_pos": int(rl.sum()), "agreement": float((cb == rl).mean()),
                 "kappa": float(cohen_kappa_score(cb, rl))})
pd.DataFrame(rows).to_csv(a.out, index=False)
print(pd.DataFrame(rows).to_string(index=False))
