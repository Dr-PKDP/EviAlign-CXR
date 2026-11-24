"""Summaries of the identical-explanation operator pass (scripts/11) and of the pair coverage of the subset passes.

Reads runs/<run>/eval_ops/ (faithfulness_<set>_{noise,blur,mean}.csv, per_image_<set>.csv, thresholds.json) and writes, to --out:
  operators_identical_explanations.csv  excess drop, win rate, normalised TGD and ECS gap (one mask, five masks) per operator
  operators_paired.csv                  blur - noise and mean - noise on the same pairs, with patient-level bootstrap intervals
  ecs_masks.csv                         ECS gap under noise infill from one mask and from five masks, with intervals
  coverage_subset.csv                   findings above the gate, after the cap, asserted, eligible and scored, from a pass directory
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SETS = [("nih_test", "NIH-14"), ("chestxdet", "ChestX-Det10"), ("rsna", "RSNA")]
SYSTEMS = [("plain_vlm", "Unguided VLM"), ("main", "EviAlign-CXR")]


def boot_ci(values, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"v": values, "g": groups}).groupby("g").v.agg(["sum", "count"])
    s, c, k = g["sum"].values, g["count"].values, len(g)
    est = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return np.percentile(est, [2.5, 97.5])


def coverage(pass_dir, subset_dir, cap_k=3):
    th = np.array(json.load(open(os.path.join(pass_dir, "thresholds.json"))))
    ts = np.array(json.load(open(os.path.join(pass_dir, "thresholds_spec.json"))))
    out = []
    for s, sn in SETS:
        p = pd.read_csv(os.path.join(pass_dir, f"per_image_{s}.csv"))
        cls = [c[2:] for c in p.columns if c.startswith("p_")]
        P = p[[f"p_{c}" for c in cls]].values
        M = p[[f"m_{c}" for c in cls]].values == 1
        gate = P >= ts[None]
        margin = P / np.maximum(ts[None], 1e-6)
        sel = np.zeros_like(gate)
        for i in range(len(P)):
            for c in sorted(np.flatnonzero(gate[i]), key=lambda c: -margin[i, c])[:cap_k]:
                sel[i, c] = True
        scored = len(pd.read_csv(os.path.join(subset_dir, f"faithfulness_{s}.csv"))) if subset_dir else np.nan
        out.append(dict(set=sn, images=len(p), above_gate=int(gate.sum()), after_cap=int(sel.sum()), asserted=int(M.sum()),
                        asserted_not_selected=int((M & ~sel).sum()), eligible=int((M & (P >= th[None])).sum()), scored=scored))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--ops_subdir", default="eval_ops", help="pass directory of the identical-explanation operator pass")
    ap.add_argument("--coverage_subdir", default="eval_noise", help="primary subset pass whose coverage is tabulated")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, pairs, ecs, cov = [], [], [], []
    for s, sn in SETS:
        for run, rn in SYSTEMS:
            d = os.path.join(a.runs, run, a.ops_subdir)
            pm = pd.read_csv(os.path.join(d, f"per_image_{s}.csv"), usecols=["idx", "patient_id"])
            pm = dict(zip(pm.idx, pm.patient_id.astype(str)))
            D = {op: pd.read_csv(os.path.join(d, f"faithfulness_{s}_{op}.csv")) for op in ("noise", "blur", "mean")}
            for op, df in D.items():
                ex = df.d_text - df.d_rand
                rows.append(dict(set=sn, system=rn, operator=op, pairs=len(df), excess_drop=ex.mean(), win_rate=(df.d_text > df.d_rand).mean(),
                                 normalised_tgd=ex.mean() / (df.d_evid - df.d_rand).mean(),
                                 ecs_gap_five_masks=(df.ecs_evid - df.ecs_rand).mean(), ecs_gap_one_mask=(df.ecs_evid - df.ecs_rand_first).mean()))
            for op in ("blur", "mean"):
                A = D[op].set_index(["img", "cls"])
                B = D["noise"].set_index(["img", "cls"])
                ix = A.index.intersection(B.index)
                diff = (A.loc[ix].d_text - A.loc[ix].d_rand) - (B.loc[ix].d_text - B.loc[ix].d_rand)
                lo, hi = boot_ci(diff.values, [pm.get(i, str(i)) for i, _ in ix], a.n_boot)
                pairs.append(dict(set=sn, system=rn, contrast=f"{op} - noise", pairs=len(ix), difference=diff.mean(), ci_low=lo, ci_high=hi))
            df = D["noise"]
            g = [pm.get(i, str(i)) for i in df.img]
            e5, e1 = df.ecs_evid - df.ecs_rand, df.ecs_evid - df.ecs_rand_first
            l5, h5 = boot_ci(e5.values, g, a.n_boot)
            l1, h1 = boot_ci(e1.values, g, a.n_boot)
            ecs.append(dict(set=sn, system=rn, pairs=len(df), ecs_one_mask=e1.mean(), ci_one=f"{l1:.4f}-{h1:.4f}", ecs_five_masks=e5.mean(),
                            ci_five=f"{l5:.4f}-{h5:.4f}", sd_random_term_across_masks=df.ecs_rand_sd.mean()))
    for run, rn in SYSTEMS:
        c = coverage(os.path.join(a.runs, run, a.coverage_subdir), os.path.join(a.runs, run, a.coverage_subdir))
        c.insert(0, "system", rn)
        cov.append(c)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "operators_identical_explanations.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(a.out, "operators_paired.csv"), index=False)
    pd.DataFrame(ecs).to_csv(os.path.join(a.out, "ecs_masks.csv"), index=False)
    pd.concat(cov).to_csv(os.path.join(a.out, "coverage_subset.csv"), index=False)


if __name__ == "__main__":
    main()
