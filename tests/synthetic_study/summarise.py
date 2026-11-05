"""Aggregate the controlled synthetic study over seeds (mean ± sd) into a CSV/Markdown table."""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = "/tmp/evialign_study"
rows = []
for run in sorted(glob.glob(f"{D}/runs/*_s*/eval/summary.json")):
    name = run.split("/")[-3]
    var, seed = name.rsplit("_s", 1)
    S = json.load(open(run))
    for st, d in S.items():
        f = d.get("faithfulness", {})
        loc = d.get("localisation", {})
        per = pd.read_csv(os.path.dirname(run) + f"/per_image_{st}.csv")
        man = pd.read_csv(f"{D}/data/{'external' if st == 'external' else 'train'}.csv")
        man = man[man["split"] == "test"].reset_index(drop=True).iloc[per["idx"].values]
        ff = pd.read_csv(os.path.dirname(run) + f"/faithfulness_{st}.csv")
        rows.append({"variant": var, "seed": int(seed), "set": st,
                     "macro_auroc": d["classification"]["macro"]["auroc"],
                     "effusion_auroc": roc_auc_score(man["y_pleural_effusion"], per["p_pleural_effusion"]),
                     "pec_f1": d["consistency"]["pec_macro_f1"], "halluc": d["consistency"]["hallucination_vs_gt"],
                     "js": f.get("js_text_vs_evidence"), "tgd_excess": float((ff["d_text"] - ff["d_rand"]).mean()),
                     "tgd_win": f.get("tgd_win_rate_vs_random"), "ecs_gap": float((ff["ecs_evid"] - ff["ecs_rand"]).mean()),
                     "evid_excess": float((ff["d_evid"] - ff["d_rand"]).mean()),
                     "evid_hit": loc.get("evidence", {}).get("hit"), "evid_ratio": loc.get("evidence", {}).get("mass_area_ratio"),
                     "text_hit": loc.get("text", {}).get("hit"), "text_ratio": loc.get("text", {}).get("mass_area_ratio"),
                     "gradcam_hit": loc.get("gradcam", {}).get("hit")})
df = pd.DataFrame(rows)
df.to_csv(f"{D}/study_all.csv", index=False)
g = df.groupby(["set", "variant"]).agg(["mean", "std"])
cols = [c for c in df.columns if c not in ("variant", "seed", "set")]
out = []
for (st, var), r in g.iterrows():
    out.append({"set": st, "variant": var, "n_seeds": int(df[(df.set == st) & (df.variant == var)].shape[0]),
                **{c: f"{r[(c, 'mean')]:.3f} ± {r[(c, 'std')]:.3f}" for c in cols}})
o = pd.DataFrame(out)
o.to_csv(f"{D}/study_summary.csv", index=False)
print(o.T.to_string())
