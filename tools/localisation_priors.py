"""Reference levels for the pointing game and a check of where the sentence-grounding maps peak.

  priors    pointing hit, box-mass share and mass-to-area ratio of three maps that never look at the image: the image
            centre, a central Gaussian, and the leave-one-out box prior of the finding (the average box mask of that
            finding over the other annotated images of the corpus). They are scored on exactly the (image, finding)
            pairs of the localisation files, so they can be set beside the evidence and sentence-grounding rows.
  sink      for each saved sentence-grounding map set: the share of maps whose peak is the modal token, where that
            token lies, the share of map mass on the five heaviest tokens, and the mean cosine similarity of a map to
            the corpus-mean map. A map set that peaks on one token for most images has an attention sink.

python tools/localisation_priors.py --runs runs/main runs/plain_vlm --out results/analysis
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob

import numpy as np
import pandas as pd
import yaml

from evialign.metrics import box_mask, localisation_scores


def sink_stats(T, g=16):
    """T: (n, g*g) non-negative maps."""
    T = np.nan_to_num(np.clip(T, 0, None))
    T = T[T.sum(1) > 0]
    if len(T) == 0:
        return {}
    pk = T.argmax(1)
    vals, cnt = np.unique(pk, return_counts=True)
    m = vals[cnt.argmax()]
    P = T / T.sum(1, keepdims=True)
    top5 = np.sort(P, 1)[:, -5:].sum(1)
    mu = P.mean(0)
    cos = (P @ mu) / (np.linalg.norm(P, axis=1) * np.linalg.norm(mu) + 1e-12)
    return {"maps": len(T), "distinct_peak_tokens": len(vals), "modal_peak_share": round(float(cnt.max() / len(T)), 3),
            "modal_token_row": int(m // g), "modal_token_col": int(m % g), "grid": g,
            "mass_on_top5_tokens": round(float(top5.mean()), 3), "cosine_to_mean_map": round(float(cos.mean()), 3)}


def main():
    from evialign.data import ManifestDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/main", "runs/plain_vlm"])
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--sets", default="configs/eval_sets.yaml")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    _os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    classes, size = cfg["data"]["classes"], cfg["data"]["image_size"]
    sets = {s["name"]: s for s in yaml.safe_load(open(a.sets))["sets"]}

    # ---------------------------------------------------------------- sink
    rows = []
    for r in a.runs:
        for f in sorted(glob.glob(f"{r}/eval/maps_*.npz")):
            s = _os.path.basename(f)[len("maps_"):-4]
            z = np.load(f, allow_pickle=True)
            if "text" not in z.files:
                continue
            T = np.stack([np.asarray(t, dtype=float).ravel() for t in z["text"]])
            st = sink_stats(T, int(round(T.shape[1] ** 0.5)))
            if st:
                rows.append({"run": _os.path.basename(r), "set": s, **st})
    if rows:
        pd.DataFrame(rows).to_csv(f"{a.out}/attention_sink.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- priors
    out = []
    yy, xx = np.mgrid[0:16, 0:16]
    gauss = np.exp(-(((yy - 7.5) / 4.0) ** 2 + ((xx - 7.5) / 4.0) ** 2) / 2)
    centre = np.zeros((16, 16)); centre[7:9, 7:9] = 1
    for sname in ["nih_test", "chestxdet", "rsna"]:
        if sname not in sets:
            continue
        f = f"{a.runs[0]}/eval/localisation_{sname}.csv"
        if not _os.path.exists(f):
            continue
        loc = pd.read_csv(f)
        loc = loc[loc["map"] == "evidence"][["img", "cls"]].drop_duplicates()
        s = sets[sname]
        ds = ManifestDataset(s["manifest"], classes, s.get("split"), size, s.get("root", ""), train=False,
                             target_mode="evidence", max_items=s.get("max_items"))
        masks = {}
        for img in loc.img.unique():
            bx = ds[int(img)]["boxes"].numpy()
            for c in set(bx[:, 0].astype(int).tolist()):
                if c >= 0:
                    masks[(int(img), classes[c])] = (bx[bx[:, 0] == c], box_mask(bx[bx[:, 0] == c], size))
        # leave-one-out box prior per finding, on the 16x16 map grid
        small = {k: v[1].reshape(16, size // 16, 16, size // 16).mean((1, 3)) for k, v in masks.items()}
        tot = {}
        for (img, c), m in small.items():
            tot.setdefault(c, [np.zeros((16, 16)), 0]); tot[c][0] += m; tot[c][1] += 1
        for (img, c), (cb, _) in masks.items():
            n = tot[c][1] - 1
            if n < 1:
                continue
            prior = (tot[c][0] - small[(img, c)]) / n
            for name, mp in [("centre point", centre), ("central Gaussian", gauss), ("box prior (leave one out)", prior)]:
                r_ = localisation_scores(mp, cb, size)
                if r_:
                    out.append({"set": sname, "finding": c, "map": name, "img": img, **r_})
    if out:
        d = pd.DataFrame(out)
        agg = d.groupby(["set", "finding", "map"]).agg(pairs=("hit", "size"), hit=("hit", "mean"), energy=("energy", "mean"),
                                                        mass_area_ratio=("mass_area_ratio", "median")).round(3).reset_index()
        agg.to_csv(f"{a.out}/prior_baselines.csv", index=False)
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
