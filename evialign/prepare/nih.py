"""NIH ChestX-ray14 -> manifest (external set when training on ReXGradient; training set in the
strictly-open configuration). Fully open: https://nihcc.app.box.com/v/ChestXray-NIHCC

Expected files under --root: Data_Entry_2017*.csv, BBox_List_2017.csv, train_val_list.txt, test_list.txt,
images/ (or images_*/images/) with the PNGs.

python -m evialign.prepare.nih --root $DATA/nih --out manifests/nih.csv [--chestxdet_root $DATA/chestxdet]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from .common import patient_split, write_manifest

NIH2CANON = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
             "Edema": "edema", "Effusion": "pleural_effusion", "Emphysema": "emphysema", "Fibrosis": "fibrosis",
             "Hernia": "hernia", "Infiltration": "lung_opacity", "Mass": "nodule_mass", "Nodule": "nodule_mass",
             "Pleural_Thickening": "pleural_thickening", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax"}
BOX2CANON = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Effusion": "pleural_effusion",
             "Infiltrate": "lung_opacity", "Infiltration": "lung_opacity", "Mass": "nodule_mass", "Nodule": "nodule_mass",
             "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anat_dir", default="", help="CheXmask-derived lung/heart masks (tools/chexmask_to_npz.py)")
    a = ap.parse_args()
    de = pd.read_csv(glob.glob(os.path.join(a.root, "Data_Entry_2017*.csv"))[0])
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(a.root, "**", "*.png"), recursive=True)}
    test = set(open(os.path.join(a.root, "test_list.txt")).read().split())
    df = pd.DataFrame({"image_path": de["Image Index"].map(paths), "patient_id": de["Patient ID"].astype(str),
                       "study_id": de["Image Index"], "dataset": "nih", "view": de["View Position"],
                       "orig_w": de.iloc[:, 7], "orig_h": de.iloc[:, 8]})
    labels = de["Finding Labels"].str.split("|")
    for nih, canon in NIH2CANON.items():
        col = f"y_{canon}"
        v = labels.map(lambda L: int(nih in L))
        df[col] = v if col not in df else df[col].combine(v, max)
    # official split: test_list for test; patient-level 90/10 train/val inside train_val_list
    is_test = df["study_id"].isin(test)
    tv = df[~is_test].copy()
    tv["split"] = patient_split(tv, fracs=(0.9, 0.1, 0.0)).replace({"test": "val"})
    df.loc[~is_test, "split"] = tv["split"]
    df.loc[is_test, "split"] = "test"
    bb = pd.read_csv(os.path.join(a.root, "BBox_List_2017.csv"))
    bb.columns = ["image", "label", "x", "y", "w", "h"] + list(bb.columns[6:])
    boxes = {}
    for _, r in bb.iterrows():
        if r["label"] in BOX2CANON:
            boxes.setdefault(r["image"], []).append({"finding": BOX2CANON[r["label"]], "x": float(r["x"]), "y": float(r["y"]),
                                                     "w": float(r["w"]), "h": float(r["h"])})
    df["boxes"] = df["study_id"].map(lambda k: json.dumps(boxes[k]) if k in boxes else "")
    df["report"] = ""
    if a.anat_dir:
        df["anat_mask"] = df["study_id"].map(lambda k: os.path.join(a.anat_dir, k.replace(".png", ".npz")))
        df.loc[~df["anat_mask"].map(os.path.exists), "anat_mask"] = ""
    df = df[df["image_path"].notna()]
    df = df[df["view"].isin(["PA", "AP"])]
    print("images with boxes:", (df["boxes"] != "").sum())
    write_manifest(df, a.out)


if __name__ == "__main__":
    main()
