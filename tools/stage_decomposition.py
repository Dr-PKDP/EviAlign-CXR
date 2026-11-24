"""Separates the changes between the noise-infill pass of the main faithfulness table (runs/<run>/eval_noise) and the later passes:
1 eval_noise (sequential random masks, cap of 64 pairs per batch), 2 the same pairs taken from the keyed pass eval_ops, 3 every eligible
pair of eval_ops (replaying the cap on it must return the pairs of stage 1), 4 eval_noise_full (full-validation thresholds).
Writes stage_decomposition.csv."""
import argparse, os
import numpy as np, pandas as pd

SETS = [("nih_test", "NIH-14"), ("chestxdet", "ChestX-Det10"), ("rsna", "RSNA")]
SYS = [("main", "EviAlign-CXR"), ("plain_vlm", "Unguided VLM")]


def boot(v, g, rng, n=1000):
    d = pd.DataFrame({"v": v, "g": g}).groupby("g").v.agg(["sum", "count"])
    s, c, k = d["sum"].values, d["count"].values, len(d)
    e = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return np.percentile(e, [2.5, 97.5])


def stats(df):
    return (df.d_text - df.d_rand).mean(), (df.ecs_evid - df.ecs_rand).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--cap", type=int, default=64)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for run, rn in SYS:
        for s, sn in SETS:
            pi = pd.read_csv(f"{a.runs}/{run}/eval_ops/per_image_{s}.csv")
            cls = [c[2:] for c in pi.columns if c.startswith("p_")]
            order = {c: i for i, c in enumerate(cls)}
            pm = dict(zip(pi.idx, pi.patient_id.astype(str)))
            A = pd.read_csv(f"{a.runs}/{run}/eval_noise/faithfulness_{s}.csv")
            N = pd.read_csv(f"{a.runs}/{run}/eval_ops/faithfulness_{s}_noise.csv")
            N["o"] = N.cls.map(order)
            N = N.sort_values(["img", "o"])
            N["b"] = N.img // 32
            cap = N.groupby("b", group_keys=False).apply(lambda d: d.iloc[:a.cap])
            assert set(zip(cap.img, cap.cls)) == set(zip(A.img, A.cls)), (run, s, len(cap), len(A))
            D = pd.read_csv(f"{a.runs}/{run}/eval_noise_full/faithfulness_{s}.csv")
            key = pd.MultiIndex.from_frame(A[["img", "cls"]])
            B = N.set_index(["img", "cls"]).loc[key].reset_index()
            A1 = A.set_index(["img", "cls"]).loc[key].reset_index()
            g = [pm.get(i, str(i)) for i in B.img]
            dex = (B.d_text - B.d_rand).values - (A1.d_text - A1.d_rand).values
            deg = (B.ecs_evid - B.ecs_rand).values - (A1.ecs_evid - A1.ecs_rand).values
            lo, hi = boot(dex, g, rng); l2, h2 = boot(deg, g, rng)
            eA, cA = stats(A); eB, cB = stats(B); eC, cC = stats(N); eD, cD = stats(D)
            rows.append(dict(system=rn, set=sn, nA=len(A), ecsA=cA, exA=eA, exB=eB, ecsB=cB, dex=dex.mean(), dex_lo=lo, dex_hi=hi,
                             decs=deg.mean(), decs_lo=l2, decs_hi=h2, nC=len(N), exC=eC, ecsC=cC, nD=len(D), exD=eD, ecsD=cD))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/stage_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
