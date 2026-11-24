"""Two checks on the generated sentences, from the outputs of evaluate.py (no GPU).

1. Laterality and vertical position. For every NIH-14 image with a radiologist box of finding c whose generated explanation asserts c,
   the side named in the asserted sentence (right/left) is compared with the side of the box centre (patient left = image right), and the
   words upper/lower are compared with the vertical position of the box centre by the area under the ROC curve for separating the
   boxes of 'lower' sentences from those of 'upper' sentences. NIH boxes are given in 1024 x 1024 coordinates. Intervals resample patients.
2. Single- and multi-finding sentences. The 'text' localisation map is the grounding of the sentence that asserts c; when that sentence
   also names another finding the same map is scored against the boxes of each named finding. Hit rate and evidence energy of the sentences
   that name more than one finding are compared with those that name one, within finding (difference averaged over findings with weights
   given by the number of multi-finding sentences), with patient-level bootstrap intervals.

python tools/sentence_location_checks.py --root . --runs main plain_vlm --out results/analysis
Writes laterality_zone.csv and single_vs_multi.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from evialign.labeler import extract_mentions

SETS = [("nih_test", "NIH-14"), ("chestxdet", "ChestX-Det10"), ("rsna", "RSNA")]
SIDE = re.compile(r"\b(right|left|bilateral(?:ly)?)\b", re.I)
UPPER = re.compile(r"\b(upper|apical|apex|apices)\b", re.I)
LOWER = re.compile(r"\b(lower|base|bases|basal|basilar|costophrenic)\b", re.I)


def side_of(s):
    k = {x.lower()[:5] for x in SIDE.findall(s)}
    if {"right", "left"} <= k:
        return "bilateral"
    return "right" if "right" in k else "left" if "left" in k else "bilateral" if k else None


def zone_of(s):
    u, l = bool(UPPER.search(s)), bool(LOWER.search(s))
    return "upper" if u and not l else "lower" if l and not u else None


def cluster_boot(groups, stat, n=2000, seed=0):
    """stat(weights) with patient-level multinomial weights; groups = array of patient index per row."""
    rng = np.random.default_rng(seed)
    P = groups.max() + 1
    v = [stat(np.bincount(rng.integers(0, P, P), minlength=P)[groups].astype(float)) for _ in range(n)]
    return np.nanpercentile(v, [2.5, 97.5])


def laterality(root, runs, out):
    m = pd.read_csv(os.path.join(root, "manifests/nih.csv"), low_memory=False)
    t = m[m.split == "test"].reset_index(drop=True)
    rows = []
    for run in runs:
        p = pd.read_csv(os.path.join(root, f"runs/{run}/eval/per_image_nih_test.csv"))
        pairs = []
        for i in np.flatnonzero(t.boxes.notna().values & (t.boxes.astype(str).str.startswith("[")).values):
            if i >= len(p):
                continue
            text = p.generated[i] if isinstance(p.generated[i], str) else ""
            ments = [x for x in extract_mentions(text) if x.status == 1]
            bx = json.loads(t.boxes[i])
            for f in sorted({d["finding"] for d in bx}):
                ss = [text[x.sent_start:x.sent_end] for x in ments if x.finding == f]
                if not ss:
                    continue
                fb = [d for d in bx if d["finding"] == f]
                sides = {"left" if (d["x"] + d["w"] / 2) / 1024 > 0.5 else "right" for d in fb}
                pairs.append(dict(img=i, pid=str(t.patient_id[i]), finding=f, true_side="bilateral" if len(sides) == 2 else next(iter(sides)),
                                  yc=float(np.mean([(d["y"] + d["h"] / 2) / 1024 for d in fb])), gen_side=side_of(" ".join(ss)),
                                  gen_zone=zone_of(" ".join(ss))))
        D = pd.DataFrame(pairs)
        D["g"] = pd.factorize(D.pid)[0]
        s = D[D.gen_side.isin(["left", "right"]) & D.true_side.isin(["left", "right"])]
        ok = (s.gen_side == s.true_side).values.astype(float); g = s.g.values
        f = lambda w: (w * ok).sum() / w.sum()
        lo, hi = cluster_boot(g, f)
        rows.append(dict(run=run, measure="side accuracy", n=len(s), patients=s.pid.nunique(), value=ok.mean(), lo=lo, hi=hi))
        for fd, d in s.groupby("finding"):
            rows.append(dict(run=run, measure=f"side accuracy: {fd}", n=len(d), patients=d.pid.nunique(), value=(d.gen_side == d.true_side).mean()))
        z = D[D.gen_zone.notna()]
        isl = (z.gen_zone == "lower").values; y = z.yc.values; gz = z.g.values

        def auc(w):
            a, b = w * isl, w * ~isl
            gt = (y[:, None] > y[None]).astype(float) + 0.5 * (y[:, None] == y[None])
            return (a[:, None] * b[None] * gt).sum() / (a.sum() * b.sum())
        a = auc(np.ones(len(z)))
        lo, hi = cluster_boot(gz, auc, n=500)
        rows.append(dict(run=run, measure=f"vertical position AUC, lower vs upper ({int(isl.sum())} lower, {int((~isl).sum())} upper)", n=len(z),
                         patients=z.pid.nunique(), value=a, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(out, "laterality_zone.csv"), index=False)


def single_vs_multi(root, run, out):
    rows = []
    for s, sn in SETS:
        ev = os.path.join(root, f"runs/{run}/eval")
        if not os.path.exists(f"{ev}/localisation_{s}.csv"):
            continue
        p = pd.read_csv(f"{ev}/per_image_{s}.csv", usecols=["idx", "patient_id", "generated"])
        txt, pid = dict(zip(p.idx, p.generated)), dict(zip(p.idx, p.patient_id.astype(str)))
        L = pd.read_csv(f"{ev}/localisation_{s}.csv")
        L = L[L["map"] == "text"].copy()
        k = []
        for r in L.itertuples():
            tx = txt.get(r.img)
            ms = [x for x in extract_mentions(tx if isinstance(tx, str) else "") if x.status == 1]
            mine = [x for x in ms if x.finding == r.cls]
            k.append(len({x.finding for x in ms if (x.sent_start, x.sent_end) == (mine[0].sent_start, mine[0].sent_end)}) if mine else 0)
        L["k"], L["pid"] = k, [pid[i] for i in L.img]
        d = L[L.k >= 1].copy()
        d["multi"] = (d.k > 1).astype(int)
        if d.multi.sum() < 20:
            continue
        pats = pd.factorize(d.pid)[0]; cl = pd.factorize(d.cls)[0]; P, C = pats.max() + 1, cl.max() + 1
        for col in ("hit", "energy"):
            N = np.zeros((2, P, C)); S = np.zeros((2, P, C))
            np.add.at(N, (d.multi.values, pats, cl), 1.0)
            np.add.at(S, (d.multi.values, pats, cl), d[col].values)

            def est(w):
                n, sm = np.einsum("p,kpc->kc", w, N), np.einsum("p,kpc->kc", w, S)
                ok = (n[0] > 0) & (n[1] > 0)
                return np.average(sm[1][ok] / n[1][ok] - sm[0][ok] / n[0][ok], weights=n[1][ok])
            lo, hi = np.nanpercentile([est(np.bincount(np.random.default_rng(j).integers(0, P, P), minlength=P).astype(float)) for j in range(2000)], [2.5, 97.5])
            rows.append(dict(run=run, set=sn, measure=col, multi_sentences=int(d.multi.sum()), single_sentences=int((1 - d.multi).sum()),
                             multi_mean=d[d.multi == 1][col].mean(), single_mean=d[d.multi == 0][col].mean(), within_finding_difference=est(np.ones(P)),
                             lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(out, "single_vs_multi.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs", nargs="+", default=["main", "plain_vlm"])
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    laterality(a.root, a.runs, a.out)
    single_vs_multi(a.root, "main", a.out)


if __name__ == "__main__":
    main()
