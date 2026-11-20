"""Classification metrics under one stated label policy, computed from the saved per-image probabilities (no GPU).

python tools/classification_policy.py --runs runs/main runs/plain_vlm runs/classifier_only --out results/analysis

Label values in the manifests are 1 (positive), 0 (negative, or not mentioned in the report), -1 (CheXbert 'uncertain') and
NaN (the corpus does not annotate the finding). Two policies are written for every run, corpus and finding:
  exclude   uncertain pairs are left out of every metric (primary: the pair has no reliable label);
  u_ones    uncertain pairs count as positive, the convention used to train the model.
NaN is always left out. The operating points use the thresholds the run saved (F1-optimal on the validation split).
Outputs: classification_policy.csv (per finding, macro and pooled rows) and label_counts.csv (positive, negative, uncertain
and unannotated pairs per corpus and finding).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from evialign.metrics import patient_bootstrap

PRIMARY = {"rsna": ["lung_opacity"]}      # RSNA annotates one adjudicated opacity endpoint; the pneumonia output is a cross-label transfer


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else np.nan


def ece_width(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        s = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if s.sum():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def ece_count(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b", observed=True).agg(n=("p", "size"), mp=("p", "mean"), ob=("y", "mean"))
    return float((t.n * (t.mp - t.ob).abs()).sum() / t.n.sum())


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--runs", nargs="+", default=["runs/main", "runs/plain_vlm", "runs/classifier_only"])
    ap_.add_argument("--config", default="configs/main.yaml")
    ap_.add_argument("--sets", default="configs/eval_sets.yaml")
    ap_.add_argument("--eval_dir", default="eval")
    ap_.add_argument("--out", default="results/analysis")
    ap_.add_argument("--n_boot", type=int, default=1000)
    a = ap_.parse_args()
    classes = yaml.safe_load(open(a.config))["data"]["classes"]
    sets = {m["name"]: m for m in yaml.safe_load(open(a.sets))["sets"]}
    rows, counts = [], []
    for r in a.runs:
        th_f = os.path.join(r, a.eval_dir, "thresholds.json")
        th = np.array(json.load(open(th_f))) if os.path.exists(th_f) else None
        for s, m in sets.items():
            f = os.path.join(r, a.eval_dir, f"per_image_{s}.csv")
            if not os.path.exists(f) or not os.path.exists(m["manifest"]):
                continue
            p = pd.read_csv(f)
            df = pd.read_csv(m["manifest"], low_memory=False)
            if "split" in df.columns and m.get("split"):
                df = df[df.split == m["split"]]
            df = df.reset_index(drop=True).loc[p.idx.values].reset_index(drop=True)
            groups = list(p["patient_id"].astype(str))
            ycols = [c for c in classes if f"y_{c}" in df.columns and df[f"y_{c}"].notna().any()]
            for c in ycols:
                y = df[f"y_{c}"].values.astype(float)
                if r == a.runs[0]:
                    counts.append({"set": s, "finding": c, "pairs": len(y), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
                                   "uncertain": int((y == -1).sum()), "unannotated": int(np.isnan(y).sum())})
            for policy in ("exclude", "u_ones"):
                perf, store = {}, {}
                for c in ycols:
                    y = df[f"y_{c}"].values.astype(float)
                    prob = p[f"p_{c}"].values
                    if policy == "exclude":
                        keep = (y == 0) | (y == 1)
                    else:
                        keep = (y == 0) | (y == 1) | (y == -1)
                        y = np.where(y == -1, 1.0, y)
                    store[c] = (keep, y, prob)
                    yy, pp = y[keep], prob[keep]
                    if keep.sum() == 0 or len(np.unique(yy)) < 2:
                        continue
                    row = {"run": os.path.basename(r), "set": s, "policy": policy, "finding": c, "n": int(keep.sum()),
                           "positives": int(yy.sum()), "prevalence": float(yy.mean()), "mean_probability": float(pp.mean()),
                           "auroc": auc(yy, pp), "auprc": ap(yy, pp), "brier": float(((pp - yy) ** 2).mean()),
                           "ece15": ece_width(yy, pp, 15)}
                    if th is not None:
                        k = classes.index(c)
                        pr = pp >= th[k]
                        tp, fp, fn = int((pr & (yy == 1)).sum()), int((pr & (yy == 0)).sum()), int((~pr & (yy == 1)).sum())
                        tn = int((~pr & (yy == 0)).sum())
                        row.update({"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "precision": tp / max(tp + fp, 1),
                                    "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
                    rows.append(row)
                    perf[c] = row
                if not perf:
                    continue
                # macro and pooled rows, with patient-level bootstrap intervals for the AUROC and AUPRC of every finding and the macro
                names = list(perf)
                boots = {c: {"auroc": [], "auprc": []} for c in names}
                boots["macro"] = {"auroc": [], "auprc": []}
                prim = [c for c in names if c in PRIMARY.get(s, names)] or names
                boots["macro_all"] = {"auroc": [], "auprc": []}
                pooled_ece15, pooled_ece10 = [], []
                for idx in patient_bootstrap(groups, a.n_boot):
                    au, apr = {}, {}
                    for c in names:
                        keep, y, prob = store[c]
                        kk = keep[idx]
                        yy, pp = y[idx][kk], prob[idx][kk]
                        au[c], apr[c] = auc(yy, pp), ap(yy, pp)
                        boots[c]["auroc"].append(au[c]); boots[c]["auprc"].append(apr[c])
                    for key, sel in (("macro", prim), ("macro_all", names)):
                        boots[key]["auroc"].append(np.nanmean([au[c] for c in sel])); boots[key]["auprc"].append(np.nanmean([apr[c] for c in sel]))
                for key, vals in boots.items():
                    ci = {}
                    for met in ("auroc", "auprc"):
                        lo, hi = np.nanpercentile(vals[met], [2.5, 97.5])
                        ci[f"{met}_lo"], ci[f"{met}_hi"] = float(lo), float(hi)
                    if key in perf:
                        perf[key].update(ci)
                    else:
                        rows.append({"run": os.path.basename(r), "set": s, "policy": policy, "finding": key, **ci})
                # point estimates for the macro rows and the pooled calibration
                for key, sel in (("macro", prim), ("macro_all", names)):
                    rec = next(x for x in rows if x["run"] == os.path.basename(r) and x["set"] == s and x["policy"] == policy and x["finding"] == key)
                    rec["auroc"] = float(np.nanmean([perf[c]["auroc"] for c in sel if c in perf]))
                    rec["auprc"] = float(np.nanmean([perf[c]["auprc"] for c in sel if c in perf]))
                    rec["ece15"] = float(np.mean([perf[c]["ece15"] for c in sel if c in perf]))       # class-wise ECE averaged over findings
                    rec["n"] = int(sum(perf[c]["n"] for c in sel if c in perf))
                    rec["findings"] = ",".join(sel)
                pp = np.concatenate([store[c][2][store[c][0]] for c in names]); yy = np.concatenate([store[c][1][store[c][0]] for c in names])
                rows.append({"run": os.path.basename(r), "set": s, "policy": policy, "finding": "pooled", "n": len(yy), "positives": int(yy.sum()),
                             "prevalence": float(yy.mean()), "mean_probability": float(pp.mean()), "brier": float(((pp - yy) ** 2).mean()),
                             "ece15": ece_width(yy, pp, 15), "ece10_equal_count": ece_count(yy, pp, 10)})
            print("done", r, s, flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "classification_policy.csv"), index=False)
    pd.DataFrame(counts).drop_duplicates(["set", "finding"]).to_csv(os.path.join(a.out, "label_counts.csv"), index=False)


if __name__ == "__main__":
    main()
