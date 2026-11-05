"""Calibrate the zero-shot frontal check on images with known view labels and inspect the inferred-frontal ones.
Prints accuracy on labelled PA/AP vs LATERAL images and the p_frontal distribution of inferred images, then writes a
contact sheet (labelled PA, labelled LATERAL, inferred) for visual inspection.

python tools/view_check.py --root $DATA/rexgradient --img_root $DATA/rexgradient_img --out view_check.jpg"""
import argparse, json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evialign.prepare.rexgradient import pick_frontal, LATERAL
from tools.view_filter import frontal_probability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--img_root", required=True)
    ap.add_argument("--out", default="view_check.jpg"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    j = json.load(open(os.path.join(a.root, "metadata", "train_metadata_view_position.json")))
    pa, lat, inf, info = [], [], [], {}
    for e in j.values():
        views = [str(x).strip().upper() for x in (e.get("ImageViewPosition") or [])]
        imgs = e.get("ImagePath") or []
        P = lambda i: os.path.join(a.img_root, imgs[i].lstrip("./").replace("deid_png", "deid_png", 1) if imgs[i].startswith("../") is False else imgs[i][3:])
        i, v, ii = pick_frontal(views)
        if i is None or i >= len(imgs): continue
        if ii: inf.append(P(i)); info[P(i)] = (e.get("Manufacturer"), tuple(views))
        else:
            pa.append(P(i))
            for k, x in enumerate(views):
                if x in LATERAL and k < len(imgs): lat.append(P(k))
    for L in (pa, lat, inf): rng.shuffle(L)
    pa, lat, inf = pa[:a.n], lat[:a.n], inf[:a.n]
    pp, pl, pi = (frontal_probability(L) for L in (pa, lat, inf))
    print(f"labelled PA/AP : n={len(pa)}  predicted frontal at 0.5 = {(pp >= .5).mean():.3f}  median p={np.median(pp):.3f}")
    print(f"labelled LATERAL: n={len(lat)}  predicted frontal at 0.5 = {(pl >= .5).mean():.3f}  median p={np.median(pl):.3f}")
    print(f"inferred        : n={len(inf)}  predicted frontal at 0.5 = {(pi >= .5).mean():.3f}  median p={np.median(pi):.3f}")
    rows = [("PA/AP", pa[:10], pp[:10]), ("LATERAL", lat[:10], pl[:10]), ("INFERRED", inf[:20], pi[:20])]
    T, cols = 200, 10
    nrow = sum(-(-len(r[1]) // cols) for r in rows)
    sheet = Image.new("L", (cols * T, nrow * (T + 14)), 255); d = ImageDraw.Draw(sheet); y = 0
    for name, ps, pr in rows:
        for s in range(0, len(ps), cols):
            for k, (p, q) in enumerate(zip(ps[s:s + cols], pr[s:s + cols])):
                im = Image.open(p).convert("L"); im.thumbnail((T, T))
                sheet.paste(im, (k * T, y)); d.text((k * T + 2, y + T), f"{name} {q:.2f}", fill=0)
            y += T + 14
    sheet.save(a.out, quality=85); print("wrote", a.out)
    if inf:
        from collections import Counter
        print("inferred-image manufacturer/view patterns:", Counter(str(info[p]) for p in inf).most_common(5))


if __name__ == "__main__":
    main()
