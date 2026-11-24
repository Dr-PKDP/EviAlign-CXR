"""Model-parameter randomisation test (Adebayo et al., 2018) for the evidence maps.
Spearman correlation between the evidence map of the trained model and that of the same model with (i) the evidence head and
(ii) the head plus the last k encoder blocks re-initialised, per image, for the class with the highest predicted probability.
A correlation near zero means the maps depend on the learned weights. The result is the median and interquartile range of the
per-image correlations, pooled over --seeds random re-initialisations. Besides the original statistic (class re-chosen by the
re-initialised model) the output holds variants that hold the class fixed, drop the rectification, use absolute values, and two
controls (random against random re-initialisation, and the trained map of a different image).

python tools/randomisation_check.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 256 --out results/randomisation.csv
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from evialign.data import Collator, ManifestDataset
from evialign.models.model import EviAlign


def reinit(model, k: int, seed: int):
    """Re-initialise the evidence head and the last k encoder blocks in place."""
    torch.manual_seed(seed)
    for p in model.head.modules():
        if hasattr(p, "reset_parameters"):
            p.reset_parameters()
    torch.nn.init.normal_(model.head.q, std=model.head.d ** -0.5)
    torch.nn.init.normal_(model.head.w, std=model.head.d ** -0.5)
    torch.nn.init.zeros_(model.head.b)
    blocks = list(getattr(model.encoder.net, "blocks", []))
    for blk in (blocks[len(blocks) - k:] if k else []):
        for p in blk.modules():
            if hasattr(p, "reset_parameters"):
                p.reset_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["model"]["decoder"] = {"type": "none"}          # the evidence maps do not involve the decoder
    base = EviAlign(cfg, len(cfg["data"]["classes"]))
    base.load_state_dict(ck["model"], strict=False)
    base.to(dev).eval()
    ds = ManifestDataset(a.manifest, cfg["data"]["classes"], a.split, cfg["data"]["image_size"], max_items=a.n)
    dl = DataLoader(ds, batch_size=16, collate_fn=Collator(None, cfg["data"]["classes"]))

    @torch.no_grad()
    def run(m):
        lg_all, phi_all = [], []
        for b in dl:
            lg, phi, _ = m.classify(m.encode(b["image"].to(dev)))
            lg_all.append(lg.float().cpu().numpy()); phi_all.append(phi.float().cpu().numpy())
        return np.concatenate(lg_all), np.concatenate(phi_all)

    def rho_of(x, y):
        return np.array([spearmanr(u, v).correlation for u, v in zip(x, y)], dtype=float)

    lg_b, phi_b = run(base)
    ix = np.arange(len(lg_b))
    c_ref = lg_b.argmax(1)                                   # class of the trained model, fixed for every comparison
    ref = phi_b[ix, c_ref]                                   # signed evidence of that class, N x M
    rows = []

    def add(name, k, variant, rho):
        rho = np.asarray(rho, dtype=float)
        q1, med, q3 = np.nanpercentile(rho, [25, 50, 75])
        rows.append({"reinitialised": name, "k_blocks": k, "variant": variant, "median_spearman": med, "q1": q1, "q3": q3,
                     "n_values": int(np.isfinite(rho).sum()), "images": len(ref), "seeds": a.seeds})
        print(f"{name:34s} {variant:28s} median {med:+.3f} (IQR {q1:+.3f} to {q3:+.3f}, {rows[-1]['n_values']} values)", flush=True)

    add("none (trained map of another image)", -1, "rectified, fixed class", rho_of(np.maximum(ref, 0), np.roll(np.maximum(ref, 0), 1, axis=0)))
    for k in a.blocks:
        name = "evidence head" if k == 0 else f"head and last {k} encoder blocks"
        per = {"as reported (argmax of re-initialised model)": [], "rectified, fixed class": [], "signed, fixed class": [],
               "absolute, fixed class": [], "random vs random, rectified, fixed class": []}
        fixed_maps, changed = [], []
        for seed in range(a.seeds):
            m = reinit(copy.deepcopy(base), k, seed)
            lg, phi = run(m.to(dev).eval())
            own = phi[ix, lg.argmax(1)]
            fx = phi[ix, c_ref]
            per["as reported (argmax of re-initialised model)"] += list(rho_of(np.maximum(ref, 0), np.maximum(own, 0)))
            per["rectified, fixed class"] += list(rho_of(np.maximum(ref, 0), np.maximum(fx, 0)))
            per["signed, fixed class"] += list(rho_of(ref, fx))
            per["absolute, fixed class"] += list(rho_of(np.abs(ref), np.abs(fx)))
            fixed_maps.append(np.maximum(fx, 0)); changed.append(float((lg.argmax(1) != c_ref).mean()))
        for s0 in range(a.seeds):
            per["random vs random, rectified, fixed class"] += list(rho_of(fixed_maps[s0], fixed_maps[(s0 + 1) % a.seeds]))
        print(f"{name}: share of images whose top class changed after re-initialisation = {np.mean(changed):.2f}", flush=True)
        for variant, rho in per.items():
            add(name, k, variant, rho)
    if a.out:
        _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("written", a.out)


if __name__ == "__main__":
    main()
