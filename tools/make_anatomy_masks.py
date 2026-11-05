"""Lung and heart masks for every training image with the open TorchXRayVision PSPNet anatomy model
(trained on ChestX-Det anatomy; masks only, no disease labels).

python tools/make_anatomy_masks.py --manifest manifests/rexgradient.csv --root_in $DATA/rexgradient --out_dir $DATA/rexgradient_anat
Then re-run the corpus adapter with --anat_dir pointing to --out_dir.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root_in", required=True, help="common prefix of image paths (mirrored under out_dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--save_side", type=int, default=256)
    a = ap.parse_args()
    import torchxrayvision as xrv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = xrv.baseline_models.chestx_det.PSPNet().to(dev).eval()
    T = seg.targets
    li, ri, hi = T.index("Left Lung"), T.index("Right Lung"), T.index("Heart")
    paths = pd.read_csv(a.manifest)["image_path"].drop_duplicates().tolist()
    for i in range(0, len(paths), a.bs):
        chunk = paths[i:i + a.bs]
        ims, shapes = [], []
        for p in chunk:
            im = Image.open(p)
            arr = np.asarray(im, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(-1)
            arr = arr / (65535.0 if arr.max() > 255 else 255.0) * 255.0
            shapes.append(arr.shape)
            x = xrv.datasets.normalize(arr, 255)
            ims.append(torch.from_numpy(np.asarray(Image.fromarray(x).resize((512, 512), Image.BILINEAR)))[None])
        with torch.no_grad():
            pr = torch.sigmoid(seg(torch.stack(ims).to(dev)))
        for p, sh, m in zip(chunk, shapes, pr):
            h, w = sh
            s = a.save_side / max(h, w)
            size = (max(1, round(h * s)), max(1, round(w * s)))
            lungs = F.interpolate(torch.maximum(m[li], m[ri])[None, None], size=size, mode="bilinear")[0, 0] > 0.5
            heart = F.interpolate(m[hi][None, None], size=size, mode="bilinear")[0, 0] > 0.5
            dst = os.path.join(a.out_dir, os.path.relpath(p, a.root_in)).rsplit(".", 1)[0] + ".npz"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, lungs=lungs.cpu().numpy().astype(np.uint8), heart=heart.cpu().numpy().astype(np.uint8))
        if i % (a.bs * 100) == 0:
            print(i, len(paths), flush=True)


if __name__ == "__main__":
    main()
