"""Paired tests between two evaluated runs on the same set.
AUROC: paired patient-level bootstrap of the macro-AUROC difference.
TGD / ECS: paired permutation test over (image, finding) pairs present in both runs.
python tools/compare_runs.py --a runs/main/eval --b runs/plain_vlm/eval --set nih_test --manifest manifests/nih.csv"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import pandas as pd

from evialign.metrics import _auc, patient_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()
pa = pd.read_csv(f"{a.a}/per_image_{a.set}.csv").sort_values("idx")
pb = pd.read_csv(f"{a.b}/per_image_{a.set}.csv").sort_values("idx")
man = pd.read_csv(a.manifest, low_memory=False)
man = man[man["split"] == a.split].reset_index(drop=True).iloc[pa["idx"].values]
cls = [c[2:] for c in pa.columns if c.startswith("p_") and f"y_{c[2:]}" in man.columns and man[f"y_{c[2:]}"].notna().any()]
Y = man[[f"y_{c}" for c in cls]].replace(-1, 1).values
out = {}


def macro(idx, P):
    v = []
    for k in range(len(cls)):
        m = ~np.isnan(Y[idx, k])
        v.append(_auc(Y[idx, k][m], P[idx, k][m]))
    return np.nanmean(v)


PA, PB = pa[[f"p_{c}" for c in cls]].values, pb[[f"p_{c}" for c in cls]].values
allidx = np.arange(len(pa))
d0 = macro(allidx, PA) - macro(allidx, PB)
ds = [macro(i, PA) - macro(i, PB) for i in patient_bootstrap(pa["patient_id"].astype(str).tolist(), a.n)]
out["macro_auroc_diff"] = {"diff": d0, "ci": list(np.nanpercentile(ds, [2.5, 97.5])), "p_two_sided": float(2 * min(np.mean(np.array(ds) <= 0), np.mean(np.array(ds) >= 0)))}
try:
    fa = pd.read_csv(f"{a.a}/faithfulness_{a.set}.csv")
    fb = pd.read_csv(f"{a.b}/faithfulness_{a.set}.csv")
    j = fa.merge(fb, on=["img", "cls"], suffixes=("_a", "_b"))
    rng = np.random.default_rng(0)
    for k in ["d_text", "ecs_evid"]:
        base = "d_rand" if k == "d_text" else "ecs_rand"
        da = (j[f"{k}_a"] - j[f"{base}_a"]).values
        db = (j[f"{k}_b"] - j[f"{base}_b"]).values
        diff = da - db
        obs = diff.mean()
        perm = [(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(5000)]
        out[k + "_minus_random"] = {"a": da.mean(), "b": db.mean(), "n_pairs": len(diff), "p": float(np.mean(np.abs(perm) >= abs(obs)))}
except FileNotFoundError:
    pass
print(json.dumps(out, indent=2, default=float))
