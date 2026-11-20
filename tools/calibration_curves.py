"""Reliability data for the classifier on every corpus that annotates findings (no GPU).

python tools/calibration_curves.py --runs runs/main --out results/analysis
Writes calibration_bins.csv (ten equal-count bins per corpus, pooled over findings and per finding for the five most
frequent findings) and calibration_summary.csv (ECE, Brier score, mean predicted probability and prevalence per corpus).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob

import numpy as np
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/main"])
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--sets", default="configs/eval_sets.yaml")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    classes = yaml.safe_load(open(a.config))["data"]["classes"]
    mp = {m["name"]: m for m in yaml.safe_load(open(a.sets))["sets"]}
    bins, summ = [], []
    for r in a.runs:
        for f in sorted(glob.glob(f"{r}/eval/per_image_*.csv")):
            s = _os.path.basename(f)[len("per_image_"):-4]
            if s not in mp:
                continue
            p = pd.read_csv(f)
            df = pd.read_csv(mp[s]["manifest"], low_memory=False)
            if "split" in df.columns and mp[s].get("split"):
                df = df[df.split == mp[s]["split"]]
            df = df.reset_index(drop=True).loc[p.idx.values]
            P, Y = [], []
            for c in classes:
                if f"y_{c}" in df.columns:
                    y = df[f"y_{c}"].values.astype(float)
                    m = (y == 0) | (y == 1)
                    P.append((c, p[f"p_{c}"].values[m], y[m]))
            if not P:
                continue
            pp = np.concatenate([x[1] for x in P]); yy = np.concatenate([x[2] for x in P])
            groups = [("all findings", pp, yy)] + [(c, a_, b_) for c, a_, b_ in sorted(P, key=lambda t: -t[2].sum())[:5]]
            for name, pa, ya in groups:
                q = pd.qcut(pa, 10, duplicates="drop")
                t = pd.DataFrame({"p": pa, "y": ya, "bin": q}).groupby("bin", observed=True).agg(n=("p", "size"), mean_p=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
                ece = float((t.n * (t.mean_p - t.observed).abs()).sum() / t.n.sum())
                for i, row in t.iterrows():
                    bins.append({"run": _os.path.basename(r), "set": s, "group": name, "bin": i, **row.round(4).to_dict()})
                summ.append({"run": _os.path.basename(r), "set": s, "group": name, "n": len(pa), "prevalence": round(float(ya.mean()), 4),
                             "mean_probability": round(float(pa.mean()), 4), "ece_10_equal_count_bins": round(ece, 4),
                             "brier": round(float(((pa - ya) ** 2).mean()), 4)})
    _os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(bins).to_csv(f"{a.out}/calibration_bins.csv", index=False)
    d = pd.DataFrame(summ); d.to_csv(f"{a.out}/calibration_summary.csv", index=False)
    print(d[d.group == "all findings"].to_string(index=False))


if __name__ == "__main__":
    main()
