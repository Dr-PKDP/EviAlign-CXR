"""Parameter counts, peak memory and latency for the three inference modes.

python tools/profile_compute.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --out results/compute_profile.csv
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from evialign.data import Collator, ManifestDataset
from evialign.models.model import EviAlign


def count(mod):
    t = sum(p.numel() for p in mod.parameters())
    tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return t, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out", default="results/compute_profile.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; classes = cfg["data"]["classes"]
    model = EviAlign(cfg, len(classes))
    model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    ds = ManifestDataset(a.manifest, classes, a.split, cfg["data"]["image_size"], max_items=a.n)
    dl = DataLoader(ds, batch_size=a.batch_size, collate_fn=Collator(None, classes))
    rows = []
    for name, mod in [("encoder", model.encoder), ("evidence head", model.head), ("decoder", model.decoder)]:
        if mod is not None:
            t, tr = count(mod)
            rows.append({"item": name, "parameters_M": round(t / 1e6, 1), "trainable_M": round(tr / 1e6, 1)})
    t, tr = count(model)
    rows.append({"item": "whole model", "parameters_M": round(t / 1e6, 1), "trainable_M": round(tr / 1e6, 1)})

    def timed(label, fn):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n, t0 = 0, None
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(dev)
                if i == 1:                      # first batch warms up
                    torch.cuda.synchronize(); t0 = time.time()
                fn(x)
                if t0 is not None:
                    n += x.shape[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        rows.append({"item": label, "images": n, "seconds_per_image": round(dt / max(1, n), 3),
                     "images_per_minute": round(60 * n / dt, 1),
                     "peak_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})

    timed("classification and evidence only", lambda x: model.classify(model.encode(x)))
    if model.decoder is not None:
        def free(x):
            z = model.encode(x)
            model.classify(z)
            model.decoder.generate_ids(model.memory(z), max_new_tokens=a.max_new_tokens)
        timed("free generation", free)

        def grounded(x):
            z = model.encode(x)
            model.classify(z)
            mem = model.memory(z)
            ids, attn = model.decoder.generate_ids(mem, max_new_tokens=a.max_new_tokens)
            model.decoder.ground(mem, ids, attn)
        timed("generation with token grounding", grounded)
    out = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "| torch", torch.__version__)


if __name__ == "__main__":
    main()
