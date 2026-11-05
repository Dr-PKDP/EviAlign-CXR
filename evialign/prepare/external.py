"""Adapters for the external validation corpora.

python -m evialign.prepare.external chestxdet  --root $DATA/chestxdet  --out manifests/chestxdet.csv
python -m evialign.prepare.external iu         --root $DATA/iu         --out manifests/iu.csv --chexbert --view_filter [--view_probe manifests/view_probe.joblib]
python -m evialign.prepare.external shenzhen   --root $DATA/shenzhen   --out manifests/shenzhen.csv
python -m evialign.prepare.external montgomery --root $DATA/montgomery --out manifests/montgomery.csv
python -m evialign.prepare.external tbx11k     --root $DATA/TBX11K     --out manifests/tbx11k.csv
python -m evialign.prepare.external rsna       --root $DATA/rsna       --out manifests/rsna.csv
python -m evialign.prepare.external vindr      --root $DATA/vinbigdata --out manifests/vindr.csv   (click-through licence)

All rows are written with split='test'. Findings a corpus does not annotate are left empty so they are
excluded from evaluation; 'y_abnormal' marks normal-vs-abnormal targets for the TB corpora, and boxes with
finding 'abnormal' are evaluated against the probability-weighted aggregate evidence map.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .common import dicom_to_png, write_manifest


def chestxdet(a):
    # ChestX-Det10 annotates ten categories (Atelectasis, Calcification, Consolidation, Effusion, Emphysema, Fibrosis, Fracture,
    # Mass, Nodule, Pneumothorax). Findings it does not annotate (cardiomegaly, pleural thickening, ...) stay empty and are masked.
    m = {"Atelectasis": "atelectasis", "Consolidation": "consolidation", "Effusion": "pleural_effusion", "Emphysema": "emphysema",
         "Fibrosis": "fibrosis", "Mass": "nodule_mass", "Nodule": "nodule_mass", "Pneumothorax": "pneumothorax"}
    rows = []
    for js in glob.glob(os.path.join(a.root, "*.json")):
        imgdir = os.path.join(a.root, "test" if "test" in js.lower() else "train")
        for r in json.load(open(js)):
            fp = os.path.join(imgdir, r["file_name"])
            if not os.path.exists(fp):
                hits = glob.glob(os.path.join(a.root, "**", r["file_name"]), recursive=True)
                fp = hits[0] if hits else fp
            row = {"image_path": fp, "patient_id": r["file_name"], "study_id": r["file_name"], "dataset": "chestxdet", "split": "test"}
            for c in set(m.values()):
                row[f"y_{c}"] = 0
            bx = []
            for s, b in zip(r.get("syms", []), r.get("boxes", [])):
                if s in m:
                    row[f"y_{m[s]}"] = 1
                    bx.append({"finding": m[s], "x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1]})
            row["boxes"] = json.dumps(bx) if bx else ""
            rows.append(row)
    return pd.DataFrame(rows)


def iu(a):
    pngs = {os.path.basename(p).rsplit(".", 1)[0]: p for p in glob.glob(os.path.join(a.root, "**", "*.png"), recursive=True)}
    rows = []
    for x in glob.glob(os.path.join(a.root, "**", "*.xml"), recursive=True):
        t = ET.parse(x).getroot()
        sec = {e.get("Label", "").upper(): (e.text or "").strip() for e in t.iter("AbstractText")}
        rid = os.path.basename(x).rsplit(".", 1)[0]
        for pi in t.iter("parentImage"):
            iid = pi.get("id")
            if iid in pngs:
                rows.append({"image_path": pngs[iid], "patient_id": rid, "study_id": rid, "dataset": "iu", "split": "test",
                             "report": sec.get("FINDINGS", "") or sec.get("IMPRESSION", ""), "impression": sec.get("IMPRESSION", "")})
    df = pd.DataFrame(rows)
    if a.view_filter:
        # The ReXGradient probe is adapted to IU with pseudo-labels from within-report rankings (cross-fitted by report);
        # the best-scoring image of each report is kept when its adapted score is at least 0.5.
        from tools.view_filter import adapt_view_probe
        p2, p1, st = adapt_view_probe(a.view_probe, df["image_path"].tolist(), df["study_id"].tolist())
        df["p_frontal_src"], df["p_frontal"] = p1, p2
        best = df.groupby("study_id")["p_frontal"].idxmax()
        pick = df.loc[best]
        keep = pick["p_frontal"] >= 0.5
        suffix = pick["image_path"].map(lambda p: os.path.basename(p).rsplit("-", 1)[-1][0])
        print(f"IU view selection: {len(df)} images in {df['study_id'].nunique()} reports; adaptation {st}; "
              f"reports with a frontal image: {int(keep.sum())}/{len(pick)}; "
              f"picked image carries the first sequence number in {float((suffix == '1').mean()):.3f} of picks")
        df = pick[keep].reset_index(drop=True)
    text = (df["report"].fillna("") + " " + df["impression"].fillna("")).tolist()
    if a.chexbert:
        import torch
        from .chexbert import label_texts
        lab = label_texts(text, "cuda" if torch.cuda.is_available() else "cpu")
        df = pd.concat([df, lab], axis=1)
    from ..labeler import label_report
    extra = ["emphysema", "fibrosis", "pleural_thickening", "hernia"]
    rl = [label_report(t, extra) for t in text]
    for j, c in enumerate(extra):
        df[f"y_{c}"] = [v[j] for v in rl]
    return df


def _tb_simple(a, prefix, name):
    rows = []
    for p in glob.glob(os.path.join(a.root, "**", f"{prefix}_*.png"), recursive=True):
        lab = int(re.search(r"_(\d)\.png$", p).group(1))
        rows.append({"image_path": p, "patient_id": os.path.basename(p), "study_id": os.path.basename(p), "dataset": name,
                     "split": "test", "y_abnormal": lab})
    return pd.DataFrame(rows)


def tbx11k(a):
    rows = []
    js = glob.glob(os.path.join(a.root, "**", "TBX11K_val.json"), recursive=True)
    boxes = {}
    if js:
        coco = json.load(open(js[0]))
        id2f = {im["id"]: im["file_name"] for im in coco["images"]}
        for an in coco["annotations"]:
            x, y, w, h = an["bbox"]
            boxes.setdefault(os.path.basename(id2f[an["image_id"]]), []).append({"finding": "abnormal", "x": x, "y": y, "w": w, "h": h})
    lst = glob.glob(os.path.join(a.root, "**", "TBX11K_val.txt"), recursive=True)
    names = open(lst[0]).read().split() if lst else []
    for n in names:
        p = os.path.join(a.root, "imgs", n)
        cat = n.split("/")[0]
        rows.append({"image_path": p, "patient_id": n, "study_id": n, "dataset": "tbx11k", "split": "test",
                     "y_abnormal": int(cat in ("sick", "tb")), "tb": int(cat == "tb"),
                     "boxes": json.dumps(boxes[os.path.basename(n)]) if os.path.basename(n) in boxes else ""})
    return pd.DataFrame(rows)


def _rsna_convert(args):
    src, dst = args
    if not os.path.exists(dst):
        from .common import dicom_to_png
        try:
            dicom_to_png(src, dst, max_side=1024)
        except Exception as e:  # unreadable DICOM
            return f"{src}: {e}"
    return None


def rsna(a):
    """RSNA Pneumonia Detection Challenge, adjudicated release (MD.ai export + DICOMs named by SOPInstanceUID).
    Final labels: L_o8w Normal, L_yd0 No Lung Opacity / Not Normal, L_v8n Lung Opacity (boxes). An image is positive if
    it carries at least one Lung Opacity box. Layout: <root>/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json and
    the zip extracted to <root>/dicom/. Positive images are scored for both 'pneumonia' and 'lung_opacity'."""
    from concurrent.futures import ProcessPoolExecutor
    j = json.load(open(glob.glob(os.path.join(a.root, "*annotations-adjudicated*.json"))[0]))
    ann = [x for d in j["datasets"] for x in d["annotations"] if x["labelId"] in ("L_o8w", "L_yd0", "L_v8n")]
    dcm = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(a.root, "dicom", "**", "*.dcm"), recursive=True)}
    per = {}
    for x in ann:
        e = per.setdefault(x["SOPInstanceUID"], {"pos": False, "boxes": [], "study": x["StudyInstanceUID"]})
        if x["labelId"] == "L_v8n":
            e["pos"] = True
            d = x.get("data")
            if isinstance(d, dict) and d.get("width"):
                e["boxes"].append((d["x"], d["y"], d["width"], d["height"]))
    out_dir = os.path.join(a.root, "png")
    jobs = [(dcm[k], os.path.join(out_dir, f"{k}.png")) for k in per if k in dcm]
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
        errs = [r for r in ex.map(_rsna_convert, jobs, chunksize=64) if r]
    print(f"rsna: {len(per)} labelled images, {len(jobs)} with DICOM, {len(errs)} unreadable")
    from PIL import Image
    rows = []
    for k, e in per.items():
        png = os.path.join(out_dir, f"{k}.png")
        if not os.path.exists(png):
            continue
        W, H = Image.open(png).size
        bx = []
        for (x, y, w, h) in e["boxes"]:
            for f in ("pneumonia", "lung_opacity"):
                bx.append({"finding": f, "x": x * W / 1024, "y": y * H / 1024, "w": w * W / 1024, "h": h * H / 1024})
        rows.append({"image_path": png, "patient_id": e["study"], "study_id": e["study"], "dataset": "rsna", "split": "test",
                     "y_pneumonia": int(e["pos"]), "y_lung_opacity": int(e["pos"]), "boxes": json.dumps(bx) if bx else ""})
    df = pd.DataFrame(rows)
    print(f"rsna: {len(df)} images, {int(df['y_pneumonia'].sum())} lung-opacity positive, {int((df['boxes'] != '').sum())} with boxes")
    return df


def vindr(a):
    m = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
         "Infiltration": "lung_opacity", "Lung Opacity": "lung_opacity", "Nodule/Mass": "nodule_mass",
         "Pleural effusion": "pleural_effusion", "Pleural thickening": "pleural_thickening", "Pneumothorax": "pneumothorax",
         "Pulmonary fibrosis": "fibrosis"}
    t = pd.read_csv(glob.glob(os.path.join(a.root, "**", "train.csv"), recursive=True)[0])
    out_dir = os.path.join(a.root, "png")
    rows = []
    for iid, g in t.groupby("image_id"):
        src = glob.glob(os.path.join(a.root, "**", f"{iid}.dicom"), recursive=True)
        if not src:
            continue
        dst = os.path.join(out_dir, f"{iid}.png")
        W, H = dicom_to_png(src[0], dst, max_side=1024) if not os.path.exists(dst) else (None, None)
        if W is None:
            import pydicom
            d = pydicom.dcmread(src[0], stop_before_pixels=True)
            W, H = int(d.Columns), int(d.Rows)
        from PIL import Image
        s = Image.open(dst).size[0] / W
        row = {"image_path": dst, "patient_id": iid, "study_id": iid, "dataset": "vindr", "split": "test"}
        bx = []
        for cname, canon in m.items():
            gc = g[g["class_name"] == cname]
            votes = gc["rad_id"].nunique()
            row[f"y_{canon}"] = max(row.get(f"y_{canon}", 0), int(votes >= 2))   # majority of 3 readers
            if votes >= 2:
                for r in gc.itertuples():
                    bx.append({"finding": canon, "x": r.x_min * s, "y": r.y_min * s, "w": (r.x_max - r.x_min) * s, "h": (r.y_max - r.y_min) * s})
        row["boxes"] = json.dumps(bx) if bx else ""
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["chestxdet", "iu", "shenzhen", "montgomery", "tbx11k", "rsna", "vindr"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chexbert", action="store_true")
    ap.add_argument("--view_filter", action="store_true", help="keep frontal images, using the probe fitted by prepare.rexgradient")
    ap.add_argument("--view_probe", default="manifests/view_probe.joblib")
    a = ap.parse_args()
    f = {"chestxdet": chestxdet, "iu": iu, "tbx11k": tbx11k, "rsna": rsna, "vindr": vindr,
         "shenzhen": lambda a: _tb_simple(a, "CHNCXR", "shenzhen"), "montgomery": lambda a: _tb_simple(a, "MCUCXR", "montgomery")}[a.which]
    df = f(a)
    df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)
    write_manifest(df, a.out)


if __name__ == "__main__":
    main()
