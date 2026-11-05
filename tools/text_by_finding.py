"""Per-finding comparison of classifier predictions, text assertions and reference labels (Supplementary Table S5).
Needed because thresholds fixed on ReXGradient transfer prevalence and label definitions (e.g. cardiomegaly), which distorts
pooled text metrics on other corpora.
python tools/text_by_finding.py --run runs/main --set nih_test --manifest manifests/nih.csv [--eval_subdir eval] [--out results/text_by_finding_nih.csv]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

CLS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax", "nodule_mass",
       "lung_opacity", "emphysema", "fibrosis", "pleural_thickening", "hernia"]
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--set", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--eval_subdir", default="eval"); ap.add_argument("--split", default="test"); ap.add_argument("--out", default="")
a = ap.parse_args()
d = os.path.join(a.run, a.eval_subdir)
per = pd.read_csv(os.path.join(d, f"per_image_{a.set}.csv"))
man = pd.read_csv(a.manifest, low_memory=False)
man = man[man["split"] == a.split].reset_index(drop=True)
thr = json.load(open(os.path.join(d, "thresholds.json")))
thr = dict(zip(CLS, thr)) if isinstance(thr, list) else thr.get("thresholds", thr)   # evaluate.py stores a list in class order
rows = []
for c in CLS:
    col = f"y_{c}"
    if col not in man.columns or man[col].notna().sum() == 0:
        continue
    y = man.loc[per["idx"].values, col].values.astype(float)
    ok = ~np.isnan(y); y = (np.nan_to_num(y) > 0).astype(int)
    p = per[f"p_{c}"].values; m = per[f"m_{c}"].values.astype(int); t = thr.get(c, 0.5)
    pred = (p >= t).astype(int)
    tp = lambda z: int(((z == 1) & (y == 1) & ok).sum())
    def prf(z):
        pr = tp(z) / max(int(((z == 1) & ok).sum()), 1); rc = tp(z) / max(int(((y == 1) & ok).sum()), 1)
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    pp, pr_, pf = prf(pred); tp_, tr_, tf = prf(m)
    rows.append({"finding": c, "n": int(ok.sum()), "prevalence": y[ok].mean(), "auroc": roc_auc_score(y[ok], p[ok]) if 0 < y[ok].sum() < ok.sum() else np.nan,
                 "clf_pred_rate": pred[ok].mean(), "text_assert_rate": m[ok].mean(), "clf_precision": pp, "clf_recall": pr_, "clf_f1": pf,
                 "text_precision": tp_, "text_recall": tr_, "text_f1": tf, "pec_f1": float(2 * ((pred & m)[ok].sum()) / max(pred[ok].sum() + m[ok].sum(), 1))})
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
if a.out:
    out.to_csv(a.out, index=False)
