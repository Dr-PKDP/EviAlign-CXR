"""ReXGradient-160K -> manifest (training / development corpus).

Release layout (huggingface.co/datasets/rajpurkarlab/ReXGradient-160K):
    metadata/{train,valid,test}_metadata_view_position.json   one entry per study (reports, age, ImagePath list, ImageViewPosition list)
    deid_png.part00..09                                       zstd-compressed tar of the PNGs, extracted with
                                                              cat deid_png.part0* | zstd -dc | tar -xf - -C <img_root>
ImagePath entries look like ../deid_png/<patient>/<accession>/studies/<uid>/series/<uid>/instances/<uid>.png and are
re-rooted under --img_root. One frontal image per study is kept (PA preferred, then AP; an unlabelled image next to lateral ones is inferred frontal and screened by a view probe trained on the labelled images); studies of patients younger
than --min_age years are dropped. The official train/valid/test split is used as released.

python -m evialign.prepare.rexgradient --root $DATA/rexgradient --img_root $DATA/rexgradient_img \
       --out manifests/rexgradient.csv --chexbert
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..labeler import label_report
from .common import write_manifest

FRONTAL = {"PA": "PA", "POSTERO_ANTERIOR": "PA", "AP": "AP"}
LATERAL = {"LATERAL", "LL", "LAT", "LEFT LATERAL", "RL"}
UNLABELLED = {"UNKNOWN", "N/A", "NONE", "NAN", ""}


def age_years(s) -> float:
    """'007Y' -> 7.0, '011M' -> 0.92, '002D' -> 0.005, '003W' -> 0.06; unparseable -> NaN."""
    s = str(s).strip().upper()
    if len(s) < 2 or not s[:-1].isdigit():
        return float("nan")
    n, u = int(s[:-1]), s[-1]
    return n * {"Y": 1.0, "M": 1 / 12, "W": 7 / 365.25, "D": 1 / 365.25}.get(u, float("nan"))


def pick_frontal(views):
    """Returns (index, view, inferred). Labelled PA is preferred over AP. If no image is labelled frontal, the single
    unlabelled image of a study whose remaining images are all lateral is taken as the frontal one (inferred=True)."""
    v = [str(x).strip().upper() for x in views]
    for want in ("PA", "AP"):
        for i, x in enumerate(v):
            if FRONTAL.get(x) == want:
                return i, want, False
    unl = [i for i, x in enumerate(v) if x in UNLABELLED]
    if len(unl) == 1 and all(x in LATERAL or i == unl[0] for i, x in enumerate(v)):
        return unl[0], "UNLABELLED", True
    return None, None, False


def load_split(path: str, split: str, img_root: str, min_age: float):
    with open(path) as f:
        j = json.load(f)
    rows, lat, st = [], [], {"studies": len(j), "age": 0, "no_frontal": 0, "no_report": 0}
    for key, e in j.items():
        a = age_years(e.get("PatientAge"))
        if not (a >= min_age):
            st["age"] += 1
            continue
        imgs, views = e.get("ImagePath") or [], e.get("ImageViewPosition") or []
        for k, x in enumerate(views):
            if str(x).strip().upper() in LATERAL and k < len(imgs):
                lat.append(imgs[k])
        i, v, inf = pick_frontal(views)
        if i is None or i >= len(imgs):
            st["no_frontal"] += 1
            continue
        fnd = e.get("Findings") if isinstance(e.get("Findings"), str) else ""
        imp = e.get("Impression") if isinstance(e.get("Impression"), str) else ""
        report = fnd.strip() or imp.strip()
        if not report:
            st["no_report"] += 1
            continue
        rel = imgs[i]
        while rel.startswith("../"):
            rel = rel[3:]
        man = e.get("Manufacturer")
        rows.append({"image_path": os.path.join(img_root, rel), "patient_id": e["PatientID"], "study_id": e["AccessionNumber"],
                     "dataset": "rexgradient", "split": split, "report": report, "impression": imp, "view": v, "view_inferred": int(inf),
                     "age": round(a, 1), "sex": e.get("PatientSex", ""), "site": e.get("InstitutionName", ""),
                     "manufacturer": man if isinstance(man, str) else ""})
    return rows, st, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="HF download directory (contains metadata/)")
    ap.add_argument("--img_root", required=True, help="directory the PNG archive was extracted into (contains deid_png/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_age", type=float, default=18.0)
    ap.add_argument("--max_studies", type=int, default=0, help="debug: keep only the first N studies per split")
    ap.add_argument("--view_qc", choices=["all", "inferred", "off"], default="all",
                    help="drop images the view probe scores as lateral: every kept image (all), only inferred-frontal ones, or none")
    ap.add_argument("--probe_out", default="manifests/view_probe.joblib", help="where the fitted view probe is saved (reused for IU X-ray)")
    ap.add_argument("--probe_n", type=int, default=3000, help="labelled images per view used to train the view probe")
    ap.add_argument("--frontal_thr", type=float, default=0.5)
    ap.add_argument("--chexbert", action="store_true", help="label reports with CheXbert (recommended)")
    ap.add_argument("--anat_dir", default="", help="directory of lung/heart masks from tools/make_anatomy_masks.py")
    a = ap.parse_args()

    rows, lat_all = [], []
    for name, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        p = os.path.join(a.root, "metadata", f"{name}_metadata_view_position.json")
        assert os.path.exists(p), f"missing {p}"
        r, st, lat = load_split(p, split, a.img_root, a.min_age)
        lat_all += [os.path.join(a.img_root, q[3:] if q.startswith('../') else q) for q in lat]
        if a.max_studies:
            r = r[: a.max_studies]
        print(f"{split}: {st} -> kept {len(r)}")
        rows += r
    df = pd.DataFrame(rows)

    df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)
    pats = {s: set(g.patient_id) for s, g in df.groupby("split")}
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = len(pats.get(x, set()) & pats.get(y, set()))
        print(f"patient overlap {x}/{y}: {ov}")
        assert ov == 0, "official splits share patients; re-split with evialign.prepare.common.patient_split"

    if a.view_qc != "off":
        from tools.view_filter import view_probe_frontal_probability
        rng = np.random.default_rng(0)
        fr = df.loc[df["view_inferred"] == 0, "image_path"].tolist()
        fr = [fr[i] for i in rng.permutation(len(fr))[: a.probe_n]]
        la = [lat_all[i] for i in rng.permutation(len(lat_all))[: a.probe_n]]
        la = [q for q in la if os.path.exists(q)]
        m = (df["view_inferred"] == 1) if a.view_qc == "inferred" else pd.Series(True, index=df.index)
        pf, stats, clf = view_probe_frontal_probability(fr, la, df.loc[m, "image_path"].tolist())
        print(f"view probe (BiomedCLIP embeddings + balanced logistic regression, trained on labelled views): {stats}")
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(a.probe_out)), exist_ok=True)
        joblib.dump(clf, a.probe_out)
        df["p_frontal"] = np.nan
        df.loc[m, "p_frontal"] = pf
        keep = pd.Series(True, index=df.index)
        keep[m] = pf >= a.frontal_thr
        for nm, g in (("labelled PA/AP", df["view_inferred"] == 0), ("inferred", df["view_inferred"] == 1)):
            gm = m & g
            if gm.any():
                print(f"{nm}: {int(gm.sum())} scored, {int(keep[gm].sum())} kept as frontal (p>={a.frontal_thr}); median p={float(df.loc[gm, 'p_frontal'].median()):.3f}")
        df = df[keep].reset_index(drop=True)
    exists = df["image_path"].map(os.path.exists)
    print(f"images found on disk: {exists.mean():.4f} ({int(exists.sum())}/{len(df)})")
    df = df[exists].reset_index(drop=True)

    if a.chexbert:
        from .chexbert import label_texts
        import torch
        studies = df.drop_duplicates("study_id")[["study_id", "report"]]
        lab = label_texts(studies["report"].tolist(), "cuda" if torch.cuda.is_available() else "cpu")
        lab["study_id"] = studies["study_id"].values
        df = df.merge(lab, on="study_id", how="left")
    # findings that CheXbert does not cover come from the transparent rule labeler
    extra = ["emphysema", "fibrosis", "pleural_thickening", "hernia"] + ([] if a.chexbert else
             ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax",
              "nodule_mass", "lung_opacity", "fracture"])
    rule = df["report"].map(lambda t: label_report(t, extra))
    for j, cname in enumerate(extra):
        df[f"y_{cname}"] = rule.map(lambda v: v[j])
    if a.anat_dir:
        df["anat_mask"] = df["image_path"].map(lambda p: os.path.join(a.anat_dir, os.path.relpath(p, a.img_root)).rsplit(".", 1)[0] + ".npz")
        df.loc[~df["anat_mask"].map(os.path.exists), "anat_mask"] = ""
    write_manifest(df, a.out)


if __name__ == "__main__":
    main()
