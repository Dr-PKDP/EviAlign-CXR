"""Model-parameter randomisation test (Adebayo et al., 2018) for the evidence maps:
Spearman correlation between maps of the trained model and of models with (i) the evidence head and
(ii) the head plus the last k encoder blocks re-initialised. Low correlation = maps depend on the weights.
python tools/randomisation_check.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 256"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import copy

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from evialign.data import Collator, ManifestDataset
from evialign.models.model import EviAlign

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--blocks", type=int, nargs="+", default=[0, 2, 4, 8])
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
base = EviAlign(cfg, len(cfg["data"]["classes"]))
base.load_state_dict(ck["model"], strict=False)
base.decoder = None
base.to(dev).eval()
ds = ManifestDataset(a.manifest, cfg["data"]["classes"], a.split, cfg["data"]["image_size"], max_items=a.n)
dl = DataLoader(ds, batch_size=16, collate_fn=Collator(None, cfg["data"]["classes"]))


@torch.no_grad()
def maps(m):
    out = []
    for b in dl:
        lg, phi, _ = m.classify(m.encode(b["image"].to(dev)))
        c = lg.argmax(1)
        out.append(torch.relu(phi[torch.arange(len(c)), c]).cpu().numpy())
    return np.concatenate(out)


ref = maps(base)
for k in a.blocks:
    m = copy.deepcopy(base)
    for p in m.head.modules():
        if hasattr(p, "reset_parameters"):
            p.reset_parameters()
    torch.nn.init.normal_(m.head.q, std=m.head.d ** -0.5), torch.nn.init.normal_(m.head.w, std=m.head.d ** -0.5)
    blocks = getattr(m.encoder.net, "blocks", [])
    for blk in list(blocks)[len(blocks) - k:] if k else []:
        for p in blk.modules():
            if hasattr(p, "reset_parameters"):
                p.reset_parameters()
    r = maps(m)
    rho = [spearmanr(x, y).correlation for x, y in zip(ref, r)]
    print(f"head + last {k} blocks re-initialised: median Spearman rho = {np.nanmedian(rho):.3f}")
