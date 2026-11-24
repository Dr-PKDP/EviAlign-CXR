"""Within-study consistency of view labels. For two-image studies, compares the probe score of the image labelled
frontal (or UNKNOWN next to a LATERAL) with its partner, separately for each label order.

python tools/view_pairs.py --root $DATA/rexgradient --img_root $DATA/rexgradient_img --probe manifests/view_probe.joblib"""
import argparse, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evialign.prepare.rexgradient import LATERAL, UNLABELLED, FRONTAL
from tools.view_filter import embed_images
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--img_root", required=True)
    ap.add_argument("--probe", required=True); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    rng = random.Random(0)
    j = json.load(open(os.path.join(a.root, "metadata", "train_metadata_view_position.json")))
    groups = {}
    for e in j.values():
        v = [str(x).strip().upper() for x in (e.get("ImageViewPosition") or [])]
        im = e.get("ImagePath") or []
        if len(v) != 2 or len(im) != 2: continue
        P = lambda k: os.path.join(a.img_root, im[k][3:] if im[k].startswith("../") else im[k])
        for k in (0, 1):
            if v[k] in FRONTAL and v[1 - k] in LATERAL:
                groups.setdefault(("labelled frontal", "frontal first" if k == 0 else "lateral first"), []).append((P(k), P(1 - k)))
            if v[k] in UNLABELLED and v[1 - k] in LATERAL:
                groups.setdefault(("UNKNOWN+LATERAL", "unknown first" if k == 0 else "lateral first"), []).append((P(k), P(1 - k)))
    clf = joblib.load(a.probe)
    for g, pairs in sorted(groups.items()):
        rng.shuffle(pairs); pairs = pairs[:a.n]
        f = clf.predict_proba(embed_images([p[0] for p in pairs]))[:, 1]
        l = clf.predict_proba(embed_images([p[1] for p in pairs]))[:, 1]
        print(f"{g[0]:16s} {g[1]:14s} n={len(pairs):3d}  P(first-named image scores more frontal)={np.mean(f > l):.3f}  "
              f"frontal>=0.5: {np.mean(f >= .5):.3f}  lateral<0.5: {np.mean(l < .5):.3f}")


if __name__ == "__main__":
    main()
