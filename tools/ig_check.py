"""Convergence and sign checks for the Integrated Gradients comparison of the localisation analysis.

For the first N annotated (image, finding) pairs of each corpus (the pairs of the localisation files) it reports, per step count,
  completeness error  |sum of the signed pixel attribution - (f(x) - f(baseline))| / |f(x) - f(baseline)|
  hit                 pointing game of the map the paper uses (absolute attribution pooled to the token grid), and of the signed
                      map clipped at zero, with the 15-pixel tolerance
  rank agreement      Spearman correlation of the token map with the map at the largest step count

python tools/ig_check.py --ckpt runs/main/best.pt --n 150 --out results/analysis
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

from evialign.data import ManifestDataset
from evialign.losses import gaussian_blur
from evialign.metrics import localisation_scores
from evialign.models.model import EviAlign


def ig_signed(model, x, c, steps):
    base = gaussian_blur(x, sigma=x.shape[-1] / 40)
    total = torch.zeros_like(x)
    bi = torch.arange(x.shape[0])
    for k in range(1, steps + 1):
        xi = (base + (k / steps) * (x - base)).detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            s = model.classify(model.encode(xi))[0][bi, c].float().sum()
        (g,) = torch.autograd.grad(s, xi)
        total += g
    return (x - base) * total / steps, base


def pool(t, grid):
    return F.adaptive_avg_pool2d(t, grid).flatten(1)[0].detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/main/best.pt")
    ap.add_argument("--sets", default="configs/eval_sets.yaml")
    ap.add_argument("--loc_dir", default="runs/main/eval")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--steps", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    classes, size = cfg["data"]["classes"], cfg["data"]["image_size"]
    model = EviAlign(cfg, len(classes))
    model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    sets = {s["name"]: s for s in yaml.safe_load(open(a.sets))["sets"]}
    rows = []
    for sname in ["nih_test", "chestxdet", "rsna"]:
        s = sets[sname]
        ds = ManifestDataset(s["manifest"], classes, s.get("split"), size, s.get("root", ""), train=False,
                             target_mode="evidence", max_items=s.get("max_items"))
        loc = pd.read_csv(f"{a.loc_dir}/localisation_{sname}.csv")
        loc = loc[loc["map"] == "evidence"].drop_duplicates(["img", "cls"])
        loc = loc.sample(n=min(a.n, len(loc)), random_state=0)
        for r in loc.itertuples():
            item = ds[int(r.img)]
            c = classes.index(r.cls)
            bx = item["boxes"].numpy()
            cb = bx[bx[:, 0] == c]
            x = item["image"][None].to(dev)
            maps, comp = {}, {}
            for st in a.steps:
                ig, base = ig_signed(model, x, torch.tensor([c], device=dev), st)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    f1 = model.classify(model.encode(x))[0][0, c].float()
                    f0 = model.classify(model.encode(base))[0][0, c].float()
                gap = float(f1 - f0)
                comp[st] = abs(float(ig.sum()) - gap) / max(abs(gap), 1e-6)
                g = model.encoder.grid
                maps[st] = (pool(ig.abs().sum(1, keepdim=True), g), pool(F.relu(ig.sum(1, keepdim=True)), g))
            ref = maps[a.steps[-1]][0]
            for st in a.steps:
                h_abs = localisation_scores(maps[st][0], cb, size)
                h_pos = localisation_scores(maps[st][1], cb, size) if maps[st][1].sum() > 0 else {}
                rows.append({"set": sname, "img": int(r.img), "cls": r.cls, "steps": st, "completeness_error": comp[st],
                             "hit_abs": h_abs.get("hit", np.nan), "hit_signed_positive": h_pos.get("hit", np.nan),
                             "spearman_to_max_steps": float(spearmanr(maps[st][0], ref)[0])})
        print(sname, "done", flush=True)
    d = pd.DataFrame(rows)
    _os.makedirs(a.out, exist_ok=True)
    d.to_csv(f"{a.out}/ig_check_pairs.csv", index=False)
    agg = d.groupby(["set", "steps"]).agg(pairs=("img", "size"), completeness_error_median=("completeness_error", "median"),
                                          completeness_error_mean=("completeness_error", "mean"),
                                          hit_abs=("hit_abs", "mean"), hit_signed_positive=("hit_signed_positive", "mean"),
                                          spearman_to_max_steps=("spearman_to_max_steps", "mean")).round(3).reset_index()
    agg.to_csv(f"{a.out}/ig_check.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
