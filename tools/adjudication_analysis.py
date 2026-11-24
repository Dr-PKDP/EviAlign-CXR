"""Summary of completed rating sheets from tools/adjudication_package.py, from one or several raters.

Reads <dir>/key.csv and every <dir>/ratings_<rater>.csv (columns of ratings_template.csv; the rater is taken from the file name when the
'reader' column is empty; repeated header lines and values outside the allowed set are dropped and counted):
  finding_present   yes / no / uncertain      (asked of the radiograph alone)
  sentence_X        0 wrong finding, side or zone; 1 finding correct, location missing or partly wrong; 2 correct
  highlight_X       0 not on the finding; 1 partly on it; 2 on it
  preferred         A / B / C / none
Every real case has a radiologist box of the named finding. Mismatched controls show the radiograph with the sentences and overlays of a
case with a different finding: a rater that can tell a wrong explanation from a right one gives them sentence rating 0 and low highlight ratings.
Raters whose controls received sentence rating 0 in fewer than --min_control_zero of the cases are reported but excluded from the pooled
comparisons. For each system the pooled table gives the share of real cases with a sentence rated 2, a highlight rated 2 and at least 1, and
the share preferred, averaged over the retained raters; differences between systems are paired by case with case-level bootstrap intervals.
Agreement is the mean pairwise linearly weighted kappa between raters on the ordinal ratings.

python tools/adjudication_analysis.py --dir adjudication --out results/analysis
Writes adjudication_raters.csv (per rater) and adjudication_systems.csv (pooled and per rater).
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd

SYS = ["guided", "unguided", "template"]


def load(dir_, key):
    out, bad = [], {}
    for f in sorted(glob.glob(os.path.join(dir_, "ratings_*.csv"))):
        name = os.path.basename(f)[8:-4]
        if name == "template":
            continue
        R = pd.read_csv(f, dtype=str).fillna("")
        R = R[R.case.isin(key.index)].drop_duplicates("case", keep="last")
        R["reader"] = [r if r.strip() else name for r in R.get("reader", pd.Series([""] * len(R), index=R.index))]
        n_bad = 0
        for c in R.columns:
            if c.startswith(("sentence_", "highlight_")):
                v = pd.to_numeric(R[c], errors="coerce"); n_bad += int((~v.isin([0, 1, 2])).sum()); R[c] = v.where(v.isin([0, 1, 2]))
        R["finding_present"] = R.finding_present.str.strip().str.lower().where(lambda x: x.isin(["yes", "no", "uncertain"]))
        R["preferred"] = R.preferred.str.strip().str.upper()
        bad[name] = n_bad
        out.append(R)
    return pd.concat(out), bad


def long(key, R):
    rows = []
    for r in R.itertuples():
        k = key.loc[r.case]
        for L in "ABC":
            rows.append(dict(case=r.case, reader=r.reader, system=k[f"arm_{L}"], control=int(k.get("control", 0)), present=r.finding_present,
                             sentence=getattr(r, f"sentence_{L}"), highlight=getattr(r, f"highlight_{L}"), preferred=float(str(r.preferred) == L)))
    return pd.DataFrame(rows)


def wkappa(a, b, k=3):
    O = np.zeros((k, k)); np.add.at(O, (np.asarray(a, int), np.asarray(b, int)), 1); O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0)); W = np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    return 1 - (W * O).sum() / (W * E).sum() if (W * E).sum() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="adjudication")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--min_control_zero", type=float, default=0.8)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    key = pd.read_csv(os.path.join(a.dir, "key.csv")).set_index("case")
    if "control" not in key:
        key["control"] = 0
    R, bad = load(a.dir, key)
    L = long(key, R)
    L["s2"], L["h2"], L["h1"] = (L.sentence == 2).astype(float), (L.highlight == 2).astype(float), (L.highlight >= 1).astype(float)
    L["s0"] = (L.sentence == 0).astype(float)
    real, ctl = L[L.control == 0], L[L.control == 1]
    rows = []
    for rd in sorted(L.reader.unique()):
        r, c = real[real.reader == rd], ctl[ctl.reader == rd]
        rows.append(dict(reader=rd, cases_rated=R[R.reader == rd].case.nunique(), invalid_ratings=bad.get(rd, 0),
                         present_yes_real=(R[(R.reader == rd) & (R.case.map(key.control) == 0)].finding_present == "yes").mean(),
                         control_sentence_zero=c.s0.mean(), real_sentence_two=r.s2.mean(), control_sentence_two=c.s2.mean(),
                         control_highlight_mean=c.highlight.mean(), real_highlight_mean=r.highlight.mean()))
    D = pd.DataFrame(rows); D["retained"] = D.control_sentence_zero >= a.min_control_zero
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "adjudication_raters.csv"), index=False); print(D.round(3).to_string())
    keep = set(D[D.retained].reader)
    rng = np.random.default_rng(0)
    out = []

    def table(df, label):
        for m in ("s2", "h2", "h1", "preferred"):
            P = df.groupby(["case", "system"])[m].mean().unstack()
            for s in SYS:
                out.append(dict(raters=label, measure=m, comparison=s, estimate=P[s].mean(), n_cases=int(P[s].notna().sum())))
            for x, y in (("guided", "unguided"), ("guided", "template"), ("unguided", "template")):
                d = (P[x] - P[y]).dropna().values
                if len(d) > 1:
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n_boot)]
                    out.append(dict(raters=label, measure=m, comparison=f"{x} - {y}", estimate=d.mean(), lo=np.percentile(bs, 2.5),
                                    hi=np.percentile(bs, 97.5), n_cases=len(d)))
    if keep:
        table(real[real.reader.isin(keep)], "retained raters pooled")
    for rd in sorted(L.reader.unique()):
        table(real[real.reader == rd], rd)
    readers = sorted(L.reader.unique())
    for c in ("sentence", "highlight"):
        P = L.pivot_table(index=["case", "system"], columns="reader", values=c, aggfunc="first")
        ks = [wkappa(*P[[x, y]].dropna().values.T) for x, y in itertools.combinations(readers, 2) if P[[x, y]].dropna().shape[0] > 5]
        if ks:
            out.append(dict(raters="all pairs", measure=f"mean pairwise weighted kappa, {c}", comparison=f"{len(ks)} pairs", estimate=float(np.nanmean(ks)),
                            n_cases=int(P.dropna(thresh=2).shape[0])))
    G = pd.DataFrame(out); G.to_csv(os.path.join(a.out, "adjudication_systems.csv"), index=False)
    print(G[G.raters.isin(["retained raters pooled", "all pairs"])].round(3).to_string())


if __name__ == "__main__":
    main()
