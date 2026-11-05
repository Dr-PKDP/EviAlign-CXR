"""Collect evaluate.py summaries from several runs into manuscript tables (CSV + Markdown).
python tools/make_tables.py --runs runs/main runs/plain_vlm runs/abl_no_align ... --out results/tables"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def flat(run, s, d):
    c = d.get("classification", {})
    r = {"run": os.path.basename(run.rstrip("/")), "set": s, "n": d.get("n_images"),
         "macro_auroc": c.get("macro", {}).get("auroc"), "auroc_ci": c.get("macro", {}).get("auroc_ci"),
         "macro_auprc": c.get("macro", {}).get("auprc"), "ece": d.get("ece"), "abnormality_auroc": d.get("abnormality_auroc")}
    for k, v in (d.get("consistency") or {}).items():
        r[k] = v
    for k, v in (d.get("text") or {}).items():
        r[f"text_{k}"] = v
    for k, v in (d.get("faithfulness") or {}).items():
        r[k] = v
    for m, v in (d.get("localisation") or {}).items():
        for k, x in v.items():
            r[f"loc_{m}_{k}"] = x
    for m, v in (d.get("deletion_insertion") or {}).items():
        for k, x in v.items():
            r[f"{m}_{k}"] = x
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval_subdir", default="eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, pcs = [], []
    for run in a.runs:
        p = os.path.join(run, a.eval_subdir, "summary.json")
        if not os.path.exists(p):
            print("missing", p)
            continue
        for s, d in json.load(open(p)).items():
            rows.append(flat(run, s, d))
            for cname, v in d.get("classification", {}).get("per_class", {}).items():
                pcs.append({"run": os.path.basename(run.rstrip("/")), "set": s, "class": cname, **{k: v.get(k) for k in ("n_pos", "auroc", "auroc_ci", "f1", "sens", "spec")}})
    os.makedirs(a.out, exist_ok=True)
    df, pc = pd.DataFrame(rows), pd.DataFrame(pcs)
    df.to_csv(os.path.join(a.out, "all_metrics.csv"), index=False)
    pc.to_csv(os.path.join(a.out, "per_class_auroc.csv"), index=False)
    blocks = {
        "Table_classification": ["run", "set", "n", "macro_auroc", "auroc_ci", "macro_auprc", "ece", "abnormality_auroc"],
        "Table_consistency": ["run", "set", "pec_macro_f1", "unsupported_by_classifier", "omitted_by_text", "hallucination_vs_gt", "text_clinical_macro_f1", "text_bleu4", "text_rougeL"],
        "Table_faithfulness": ["run", "set", "n_pairs", "tgd_drop_text", "tgd_drop_evidence", "tgd_drop_random", "tgd_excess_drop", "tgd_excess_drop_ci", "tgd_normalised", "tgd_win_rate_vs_random", "ecs_evidence", "ecs_random", "ecs_gap", "ecs_gap_ci", "js_text_vs_evidence"],
        "Table_localisation": ["run", "set"] + sorted([c for c in df.columns if c.startswith("loc_")]),
    }
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        for name, cols in blocks.items():
            cols = [c for c in cols if c in df.columns]
            sub = df[cols].dropna(axis=0, how="all", subset=[c for c in cols if c not in ("run", "set")])
            f.write(f"## {name}\n\n{sub.to_markdown(index=False, floatfmt='.3f')}\n\n")
            sub.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    print(open(os.path.join(a.out, "tables.md")).read())


if __name__ == "__main__":
    main()
