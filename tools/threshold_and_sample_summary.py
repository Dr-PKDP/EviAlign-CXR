"""Summaries of the passes of scripts 11 and 12: the identical-explanation pass with prefix and with full-validation thresholds
(runs/<run>/eval_ops, eval_ops_fullthr), the patient-random noise pass (eval_patients) and the keyed reduced-protocol seed passes.
Writes fullthr_passes.csv, patient_random.csv, seeds_keyed.csv and threshold_vectors.csv to --out."""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats as st

SETS = [("nih_test", "NIH-14"), ("chestxdet", "ChestX-Det10"), ("rsna", "RSNA")]
FULL = [("main", "EviAlign-CXR"), ("plain_vlm", "Unguided VLM")]
SEEDS = {"EviAlign-CXR": ["s_main", "s_main_s1", "s_main_s2"], "- alignment and instance": ["s_no_align", "s_no_align_s1", "s_no_align_s2"],
         "Unguided VLM": ["s_plain", "s_plain_s1", "s_plain_s2"]}
CLASSES = ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax", "nodule_mass",
           "lung_opacity", "emphysema", "fibrosis", "pleural_thickening", "hernia"]


def boot(v, g, rng, n=1000):
    d = pd.DataFrame({"v": np.asarray(v), "g": np.asarray(g)}).groupby("g").v.agg(["sum", "count"])
    s, c, k = d["sum"].values, d["count"].values, len(d)
    e = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return np.percentile(e, [2.5, 97.5])


def patients(d, s):
    p = pd.read_csv(f"{d}/per_image_{s}.csv", usecols=["idx", "patient_id"])
    return dict(zip(p.idx, p.patient_id.astype(str)))


def faith(d, s, op=None):
    return pd.read_csv(f"{d}/faithfulness_{s}_{op}.csv" if op else f"{d}/faithfulness_{s}.csv")


def summary(df, m, rng):
    ex, e5 = df.d_text - df.d_rand, df.ecs_evid - df.ecs_rand
    g = [m.get(i, str(i)) for i in df.img]
    lo, hi = boot(ex, g, rng); elo, ehi = boot(e5, g, rng)
    return dict(pairs=len(df), patients=len(set(g)), excess=ex.mean(), excess_lo=lo, excess_hi=hi, win=(df.d_text > df.d_rand).mean(),
                ecs5=e5.mean(), ecs5_lo=elo, ecs5_hi=ehi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run, rn in FULL:
        for s, sn in SETS:
            for thr, sub in (("prefix", "eval_ops"), ("full", "eval_ops_fullthr")):
                d = f"{a.runs}/{run}/{sub}"
                if not os.path.exists(f"{d}/DONE"):
                    continue
                m = patients(d, s)
                D = {op: faith(d, s, op) for op in ("noise", "blur", "mean")}
                for op, df in D.items():
                    rows.append(dict(system=rn, set=sn, thresholds=thr, op=op, **summary(df, m, rng)))
                A, B = D["blur"].set_index(["img", "cls"]), D["noise"].set_index(["img", "cls"])
                ix = A.index.intersection(B.index)
                dd = (A.loc[ix].d_text - A.loc[ix].d_rand) - (B.loc[ix].d_text - B.loc[ix].d_rand)
                lo, hi = boot(dd.values, [m.get(i, str(i)) for i, _ in ix], rng)
                rows.append(dict(system=rn, set=sn, thresholds=thr, op="blur_minus_noise", pairs=len(ix), excess=dd.mean(), excess_lo=lo, excess_hi=hi))
    pd.DataFrame(rows).to_csv(f"{a.out}/fullthr_passes.csv", index=False)
    rows = []
    for run, rn in FULL:
        for s, sn in SETS:
            d = f"{a.runs}/{run}/eval_patients"
            if os.path.exists(f"{d}/DONE"):
                rows.append(dict(system=rn, set=sn, **summary(faith(d, s), patients(d, s), rng)))
    pd.DataFrame(rows).to_csv(f"{a.out}/patient_random.csv", index=False)
    rows = []
    for sysn, runs in SEEDS.items():
        for seed, run in enumerate(runs):
            for s, sn in SETS:
                d = f"{a.runs}/{run}/eval_ops"
                if os.path.exists(f"{d}/DONE"):
                    rows.append(dict(system=sysn, seed=seed, set=sn, **summary(faith(d, s, "noise"), patients(d, s), rng)))
    Z = pd.DataFrame(rows)
    Z.to_csv(f"{a.out}/seeds_keyed.csv", index=False)
    rows = []
    for run, rn in FULL:
        for sub, tag in (("eval_ops", "prefix"), ("eval_ops_fullthr", "full")):
            f1 = json.load(open(f"{a.runs}/{run}/{sub}/thresholds.json")); gate = json.load(open(f"{a.runs}/{run}/{sub}/thresholds_spec.json"))
            for i, c in enumerate(CLASSES):
                rows.append(dict(system=rn, thresholds=tag, finding=c, f1_threshold=f1[i], gate=gate[i]))
    pd.DataFrame(rows).to_csv(f"{a.out}/threshold_vectors.csv", index=False)
    if len(Z):
        for s, sn in SETS:
            g = {k: Z[(Z.system == k) & (Z.set == sn)].sort_values("seed").excess.values for k in SEEDS}
            for x, y in (("EviAlign-CXR", "Unguided VLM"), ("EviAlign-CXR", "- alignment and instance"), ("- alignment and instance", "Unguided VLM")):
                print(sn, x, "-", y, round(float(np.mean(g[x] - g[y])), 3), "paired p", round(float(st.ttest_rel(g[x], g[y]).pvalue), 3))


if __name__ == "__main__":
    main()
