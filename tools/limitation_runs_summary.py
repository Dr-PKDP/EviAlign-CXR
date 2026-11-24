"""Summaries of the runs of scripts 13 and 14, written to --out:
  limitation_runs.csv          per-run values on each external corpus: noise-infill excess drop, win rate and five-mask ECS gap (keyed pass,
                               runs/<run>/eval_ops), macro-AUROC on the corpus subset, and, from the training log, the clipping statistics
  limitation_groups.csv        mean and standard deviation over seeds, and the paired differences between groups with the paired t-test p
Groups are given as --group label=run,run,run (seed 0, 1, 2). The defaults are the separate-clipping runs, the minimal-objective runs and the
full-protocol replicates of the main and unguided models, each next to its three-seed reference.

python tools/limitation_runs_summary.py --runs runs --out results/analysis
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st

SETS = [("nih_test", "NIH-14"), ("chestxdet", "ChestX-Det10"), ("rsna", "RSNA")]
REDUCED = {"EviAlign-CXR": "s_main,s_main_s1,s_main_s2", "- alignment and instance": "s_no_align,s_no_align_s1,s_no_align_s2",
           "Unguided VLM": "s_plain,s_plain_s1,s_plain_s2",
           "Separate clipping: EviAlign-CXR": "s_sep_main,s_sep_main_s1,s_sep_main_s2",
           "Separate clipping: - alignment and instance": "s_sep_no_align,s_sep_no_align_s1,s_sep_no_align_s2",
           "Separate clipping: Unguided VLM": "s_sep_plain,s_sep_plain_s1,s_sep_plain_s2",
           "Minimal objective": "s_minimal,s_minimal_s1,s_minimal_s2"}
FULL = {"Full protocol: EviAlign-CXR": "main,main_s1,main_s2", "Full protocol: Unguided VLM": "plain_vlm,plain_vlm_s1,plain_vlm_s2"}
PAIRS = [("EviAlign-CXR", "Unguided VLM"), ("EviAlign-CXR", "- alignment and instance"), ("- alignment and instance", "Unguided VLM"),
         ("Separate clipping: EviAlign-CXR", "Separate clipping: Unguided VLM"),
         ("Separate clipping: EviAlign-CXR", "Separate clipping: - alignment and instance"),
         ("Separate clipping: - alignment and instance", "Separate clipping: Unguided VLM"),
         ("Minimal objective", "Unguided VLM"), ("Minimal objective", "EviAlign-CXR"),
         ("Full protocol: EviAlign-CXR", "Full protocol: Unguided VLM")]


def run_row(root, run, ops_dir):
    d = os.path.join(root, run, ops_dir)
    if not os.path.exists(os.path.join(d, "DONE")):
        return []
    rows = []
    for s, sn in SETS:
        f = next((os.path.join(d, n) for n in (f"faithfulness_{s}_noise.csv", f"faithfulness_{s}.csv") if os.path.exists(os.path.join(d, n))), None)
        if f is None:
            continue
        f = pd.read_csv(f)
        r = dict(run=run, set=sn, pairs=len(f), excess=(f.d_text - f.d_rand).mean(), win=(f.d_text > f.d_rand).mean(),
                 ecs5=(f.ecs_evid - f.ecs_rand).mean())
        for sub in ("eval_noise", "eval"):
            p = os.path.join(root, run, sub, "summary.json")
            if os.path.exists(p) and s in json.load(open(p)):
                r["auroc"] = json.load(open(p))[s]["classification"]["macro"]["auroc"]
                break
        rows.append(r)
    log = os.path.join(root, run, "log.jsonl")
    if os.path.exists(log):
        ep = [json.loads(x) for x in open(log) if '"val_macro_auroc"' in x]
        ep = [e for e in ep if "clip_frac_cls" in e and e["epoch"] >= 2]
        if ep:
            for r in rows:
                for k in ("clip_frac_cls", "clip_frac_dec", "clip_frac_global", "global_scale_mean", "gn_cls_mean", "gn_dec_mean"):
                    r[k] = float(np.mean([e[k] for e in ep]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--group", action="append", default=[])
    a = ap.parse_args()
    groups = {k.split("=")[0]: k.split("=")[1] for k in a.group} if a.group else {**REDUCED, **FULL}
    rows = []
    for g, runs in groups.items():
        for seed, run in enumerate(runs.split(",")):
            for r in run_row(a.runs, run, "eval_ops"):
                rows.append(dict(group=g, seed=seed, **r))
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("no finished eval_ops directories found; run scripts/14_eval_limitation_runs.sh first")
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "limitation_runs.csv"), index=False)
    out = []
    metrics = [c for c in ("excess", "win", "ecs5", "auroc", "clip_frac_cls", "clip_frac_dec", "clip_frac_global", "global_scale_mean") if c in R]
    for g in groups:
        for sn in [x[1] for x in SETS]:
            d = R[(R.group == g) & (R.set == sn)]
            if len(d):
                out.append(dict(kind="mean", group=g, set=sn, n_seeds=len(d), **{m: d[m].mean() for m in metrics}, **{m + "_sd": d[m].std() for m in ("excess", "ecs5")}))
    for x, y in PAIRS:
        for sn in [s[1] for s in SETS]:
            dx, dy = R[(R.group == x) & (R.set == sn)].sort_values("seed"), R[(R.group == y) & (R.set == sn)].sort_values("seed")
            seeds = sorted(set(dx.seed) & set(dy.seed))
            if len(seeds) < 2:
                continue
            row = dict(kind="paired difference", group=f"{x} - {y}", set=sn, n_seeds=len(seeds))
            for m in ("excess", "ecs5", "win", "auroc"):
                if m in R:
                    u, v = dx.set_index("seed").loc[seeds, m].values, dy.set_index("seed").loc[seeds, m].values
                    row[m] = float(np.mean(u - v)); row[m + "_p"] = float(st.ttest_rel(u, v).pvalue) if m in ("excess", "ecs5") else np.nan
            out.append(row)
    G = pd.DataFrame(out)
    G.to_csv(os.path.join(a.out, "limitation_groups.csv"), index=False)
    print(G.round(3).to_string())


if __name__ == "__main__":
    main()
