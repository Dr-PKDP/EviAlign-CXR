"""Supplementary tables computed from existing evaluation outputs.

  tb            tuberculosis counterfactual intervals clustered by image instead of by (image, finding) pair
  ledger        gate, three-finding cap, assertion, eligibility and scoring counts for every corpus
  gate          what the 90%-specificity gate, fitted internally, actually achieves on each external corpus
  perfinding    per-finding AUROC, AUPRC, sensitivity, specificity, precision, F1 and Brier score on every corpus
  loccount      images and boxes behind each per-finding localisation score
  seeds         per-seed rows for every corpus, including RSNA

python tools/analysis_tables.py --out results/analysis
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(0)


def boot(df, stat, cluster, n=1000):
    groups = [g.index.values for _, g in df.groupby(cluster)]
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        out.append(stat(df.loc[np.concatenate([groups[k] for k in pick])]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


excess = lambda d: float((d.d_text - d.d_rand).mean())
ecsgap = lambda d: float((d.ecs_evid - d.ecs_rand).mean())


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/main", "runs/plain_vlm", "runs/classifier_only"])
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    _os.makedirs(a.out, exist_ok=True)
    classes = yaml.safe_load(open(a.config))["data"]["classes"]

    # ---------------------------------------------------------------- tuberculosis corpora, clustered by image
    rows = []
    for r in a.runs:
        for f in sorted(glob.glob(f"{r}/eval_tb/faithfulness_*.csv")):
            s = _os.path.basename(f)[len("faithfulness_"):-4]
            d = pd.read_csv(f).reset_index(drop=True)
            lo, hi = boot(d, excess, "img")
            lo2, hi2 = boot(d, ecsgap, "img")
            plo, phi = boot(d.assign(u=np.arange(len(d))), excess, "u")
            rows.append({"run": _os.path.basename(r), "set": s, "pairs": len(d), "images": d.img.nunique(),
                         "excess": round(excess(d), 3), "image_lo": round(lo, 3), "image_hi": round(hi, 3),
                         "pair_lo": round(plo, 3), "pair_hi": round(phi, 3),
                         "ecs_gap": round(ecsgap(d), 3), "ecs_lo": round(lo2, 3), "ecs_hi": round(hi2, 3)})
    if rows:
        pd.DataFrame(rows).to_csv(f"{a.out}/tb_clustered.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- ledger, gate behaviour, per-finding metrics
    led, gate, pf, lc = [], [], [], []
    for r in a.runs:
        for sub in ["eval", "eval_noise", "eval_tb"]:
            for f in sorted(glob.glob(f"{r}/{sub}/per_image_*.csv")):
              try:
                  s = _os.path.basename(f)[len("per_image_"):-4]
                  p = pd.read_csv(f)
                  if f"p_{classes[0]}" not in p.columns:
                      continue
                  P = p[[f"p_{c}" for c in classes]].values
                  M = p[[f"m_{c}" for c in classes]].values.astype(bool) if f"m_{classes[0]}" in p.columns else None
                  th = np.array(json.load(open(f"{r}/eval/thresholds.json")))
                  sp_path = f"{r}/eval/thresholds_spec.json"
                  sp = np.array(json.load(open(sp_path))) if _os.path.exists(sp_path) else None
                  man = yaml.safe_load(open("configs/eval_sets.yaml"))["sets"]
                  mp = {m["name"]: m for m in man}
                  y = None
                  if s in mp:
                      df = pd.read_csv(mp[s]["manifest"], low_memory=False)
                      if "split" in df.columns and mp[s].get("split"):
                          df = df[df.split == mp[s]["split"]]
                      df = df.reset_index(drop=True)
                      cols = [f"y_{c}" for c in classes if f"y_{c}" in df.columns]
                      if cols:
                          y = df.loc[p.idx.values, cols].values.astype(float)
                          ycls = [c.replace("y_", "") for c in cols]
                  if M is not None and sp is not None:
                      gated = P >= sp[None]
                      margin = P / np.maximum(sp[None], 1e-6)
                      cap = np.zeros_like(gated)
                      for i in range(len(P)):
                          ch = [c for c in np.argsort(-margin[i])[:3] if gated[i, c]]
                          cap[i, ch] = True
                      pred = P >= th[None]
                      fp = f"{r}/{sub}/faithfulness_{s}.csv"
                      n_scored = len(pd.read_csv(fp)) if _os.path.exists(fp) else 0
                      led.append({"run": _os.path.basename(r), "subdir": sub, "set": s, "images": len(p),
                                  "gated_before_cap": int(gated.sum()), "after_cap": int(cap.sum()),
                                  "asserted": int(M.sum()), "asserted_and_selected": int((M & cap).sum()),
                                  "asserted_not_selected": int((M & ~cap).sum()),
                                  "eligible_predicted_and_asserted": int((M & pred).sum()), "scored": n_scored,
                                  "not_scored": int((M & pred).sum()) - n_scored})
                      if y is not None and sub in ("eval",):
                          ann = (y == 0) | (y == 1)      # NaN is unannotated, -1 is uncertain; both are left out
                          ki = [classes.index(c) for c in ycls]
                          for j, c in enumerate(ycls):
                              k = ki[j]
                              m = ann[:, j]
                              if m.sum() == 0 or len(np.unique(y[m, j])) < 2:
                                  continue
                              yy, pp = y[m, j], P[m, k]
                              gsel = (pp >= sp[k])
                              gate.append({"run": _os.path.basename(r), "set": s, "finding": c, "n": int(m.sum()),
                                           "gate_threshold": round(float(sp[k]), 4),
                                           "specificity_at_gate": round(float(1 - gsel[yy == 0].mean()), 3),
                                           "sensitivity_at_gate": round(float(gsel[yy == 1].mean()), 3),
                                           "selected_rate": round(float(gsel.mean()), 3)})
                              pr = (pp >= th[k])
                              tp, fp_, fn = int((pr & (yy == 1)).sum()), int((pr & (yy == 0)).sum()), int((~pr & (yy == 1)).sum())
                              pf.append({"run": _os.path.basename(r), "set": s, "finding": c, "n": int(m.sum()),
                                         "positives": int((yy == 1).sum()),
                                         "auroc": round(float(roc_auc_score(yy, pp)), 3),
                                         "auprc": round(float(average_precision_score(yy, pp)), 3),
                                         "f1_threshold": round(float(th[k]), 4),
                                         "sensitivity": round(tp / max(1, tp + fn), 3),
                                         "specificity": round(1 - fp_ / max(1, int((yy == 0).sum())), 3),
                                         "precision": round(tp / max(1, tp + fp_), 3),
                                         "f1": round(2 * tp / max(1, 2 * tp + fp_ + fn), 3),
                                         "brier": round(float(((pp - yy) ** 2).mean()), 4)})
              except Exception as e:                                    # a corpus without usable labels must not stop the rest
                  print("skipped", r, sub, f, type(e).__name__, e)
            for f in sorted(glob.glob(f"{r}/{sub}/localisation_*.csv")):
                s = _os.path.basename(f)[len("localisation_"):-4]
                d = pd.read_csv(f)
                for (c, mn), g in d.groupby(["cls", "map"]):
                    lc.append({"run": _os.path.basename(r), "subdir": sub, "set": s, "finding": c, "map": mn,
                               "images": g.img.nunique(), "rows": len(g), "hit": round(float(g.hit.mean()), 3)})
    for name, rows_ in [("ledger", led), ("gate_external", gate), ("per_finding_metrics", pf), ("localisation_counts", lc)]:
        if rows_:
            pd.DataFrame(rows_).to_csv(f"{a.out}/{name}.csv", index=False)
            print(name, len(rows_), "rows")

    # ---------------------------------------------------------------- per-seed rows for every corpus
    seeds = []
    for r in sorted(glob.glob("runs/s_*")):
        for f in sorted(glob.glob(f"{r}/eval_noise/faithfulness_*.csv")):
            s = _os.path.basename(f)[len("faithfulness_"):-4]
            d = pd.read_csv(f)
            pi = f"{r}/eval_noise/per_image_{s}.csv"
            pat = pd.read_csv(pi)[["idx", "patient_id"]] if _os.path.exists(pi) else None
            n_pat = d.merge(pat, left_on="img", right_on="idx").patient_id.nunique() if pat is not None else None
            seeds.append({"run": _os.path.basename(r), "set": s, "pairs": len(d), "patients": n_pat,
                          "excess": round(excess(d), 3), "ecs_gap": round(ecsgap(d), 3),
                          "win_rate": round(float((d.d_text > d.d_rand).mean()), 3)})
    if seeds:
        pd.DataFrame(seeds).to_csv(f"{a.out}/seeds_all_sets.csv", index=False)
        print("seeds", len(seeds), "rows")


if __name__ == "__main__":
    main()
