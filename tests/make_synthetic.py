"""Synthetic chest-X-ray-like data with known lesion locations, reports, boxes and anatomy masks.

Used only to smoke-test the full pipeline on CPU (shapes, gradients, metrics, I/O).
python tests/make_synthetic.py --out /tmp/synth --n 600 --size 64
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

FIND = ["pleural_effusion", "consolidation", "cardiomegaly", "nodule_mass", "pneumothorax"]


def ellipse(h, w, cy, cx, ry, rx):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def make(i, S, rng, shift=0.0, marker_p=None):
    img = np.full((S, S), 40.0)
    lungR = ellipse(S, S, S * .5, S * .3, S * .32, S * .16)
    lungL = ellipse(S, S, S * .5, S * .7, S * .32, S * .16)
    lungs = lungR | lungL
    img[lungs] = 90
    labels = {f: 0 for f in FIND}
    sents, boxes = [], []
    card = rng.random() < .3
    hr = S * (.17 if card else .11)
    heart = ellipse(S, S, S * .62, S * .5, hr, hr)
    img[heart] = 170
    if card:
        labels["cardiomegaly"] = 1
        sents.append("the heart size is enlarged.")
        boxes.append({"finding": "cardiomegaly", "x": S * .5 - hr, "y": S * .62 - hr, "w": 2 * hr, "h": 2 * hr})
    for f in ["pleural_effusion", "consolidation", "nodule_mass", "pneumothorax"]:
        if rng.random() < .3:
            labels[f] = 1
            side = rng.choice(["right", "left"])
            cx = S * (.3 if side == "right" else .7)
            if f == "pleural_effusion":
                cy, r, val, txt = S * .76, S * .08, 200, f"there is a small {side} pleural effusion."
            elif f == "consolidation":
                cy, r, val = S * rng.uniform(.35, .65), S * .07, 180
                zone = "upper" if cy < S * .45 else "middle" if cy < S * .55 else "lower"
                txt = f"there is consolidation in the {side} {zone} lung zone."
            elif f == "nodule_mass":
                cy, r, val, txt = S * rng.uniform(.3, .6), S * .035, 230, f"there is a nodule in the {side} lung."
                cx += rng.uniform(-S * .05, S * .05)
            else:
                cy, r, val, txt = S * .25, S * .07, 10, f"there is a {side} pneumothorax at the apex."
            m = ellipse(S, S, cy, cx, r, r)
            img[m] = val
            sents.append(txt)
            boxes.append({"finding": f, "x": cx - r, "y": cy - r, "w": 2 * r, "h": 2 * r})
    if not sents:
        sents.append("no acute cardiopulmonary abnormality.")
    if marker_p is not None:
        # spurious cue outside the lungs: correlated with effusion in training, independent of it externally
        p_on = marker_p if labels["pleural_effusion"] else 1 - marker_p
        if rng.random() < p_on:
            img[2:8, 2:8] = 250
    img = img * (1 + shift) + rng.normal(0, 12, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, labels, " ".join(sents), boxes, lungs, heart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--shortcut", type=float, default=None, help="P(marker | effusion) in training, e.g. 0.9")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)
    rng = np.random.default_rng(0)
    for name, n, shift in [("train", a.n, 0.0), ("external", a.n // 3, 0.25)]:
        rows = []
        for i in range(n):
            mp_ = None if a.shortcut is None else (a.shortcut if name == "train" else 0.5)
            img, lab, rep, bx, lungs, heart = make(i, a.size, rng, shift, mp_)
            p = f"{a.out}/img/{name}_{i}.png"
            Image.fromarray(img).save(p)
            mp = f"{a.out}/img/{name}_{i}_anat.npz"
            np.savez_compressed(mp, lungs=lungs.astype(np.uint8), heart=heart.astype(np.uint8))
            split = "test" if name == "external" else ("train" if i < .7 * n else "val" if i < .85 * n else "test")
            r = {"image_path": p, "patient_id": f"{name}{i // 2}", "study_id": f"{name}{i}", "dataset": f"synth_{name}",
                 "split": split, "report": rep, "boxes": json.dumps(bx), "anat_mask": mp}
            r.update({f"y_{k}": v for k, v in lab.items()})
            rows.append(r)
        import pandas as pd
        pd.DataFrame(rows).to_csv(f"{a.out}/{name}.csv", index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
