"""Recompute the rule-labeler columns (emphysema, fibrosis, pleural_thickening, hernia) of a manifest from its report text.
Needed after the finding patterns change; CheXbert columns are left untouched.
python tools/relabel_rule.py manifests/rexgradient.csv manifests/iu.csv"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
import pandas as pd
from evialign.labeler import label_report

EXTRA = ["emphysema", "fibrosis", "pleural_thickening", "hernia"]
for p in sys.argv[1:]:
    df = pd.read_csv(p, low_memory=False)
    # IU labels were produced from findings + impression (see prepare.external.iu); the other corpora use the report column
    both = "impression" in df and (df.get("dataset", pd.Series([""])).astype(str) == "iu").all()
    texts = (df["report"].fillna("") + " " + df["impression"].fillna("")) if both else df["report"]
    rule = [label_report(t if isinstance(t, str) else "", EXTRA) for t in texts]
    for j, c in enumerate(EXTRA):
        old = int((df[f"y_{c}"] == 1).sum()) if f"y_{c}" in df else -1
        df[f"y_{c}"] = [v[j] for v in rule]
        print(f"{_os.path.basename(p)}: {c} positives {old} -> {int((df[f'y_{c}'] == 1).sum())}")
    df.to_csv(p, index=False)
