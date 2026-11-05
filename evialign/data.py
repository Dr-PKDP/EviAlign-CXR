"""Manifest-based dataset shared by all corpora.

Every dataset adapter in ``evialign/prepare`` writes a CSV manifest with columns
    image_path, patient_id, study_id, dataset, split, report (may be empty),
    y_<finding> for each canonical finding (1 / 0 / -1 uncertain / empty = not annotated),
    boxes (JSON list of {"finding", "x", "y", "w", "h"} in original pixels; may be empty),
    anat_mask (path to an .npz with uint8 arrays "lungs" and "heart" in original pixels; may be empty).
Unannotated findings are masked out of the classification loss and of evaluation.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .labeler import extract_mentions, is_normal, positive_sentence_spans

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Pad to square (zero) then resize to ``size``; returns image, scale, pad_x, pad_y."""
    h, w = img.shape[:2]
    s = max(h, w)
    pad_y, pad_x = (s - h) // 2, (s - w) // 2
    canvas = np.zeros((s, s), dtype=img.dtype)
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = img
    out = np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))
    return out, size / s, pad_x, pad_y


def _load_gray(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return a.astype(np.uint8)
    return np.asarray(im.convert("L"))


def evidence_sentences(report: str, classes: list[str]) -> str:
    """Explanation target = sentences asserting a finding (+ a normal statement if there are none)."""
    spans = positive_sentence_spans(report, classes)
    keep = sorted({sp for v in spans.values() for sp in v})
    if keep:
        return " ".join(report[s:e].strip() for s, e in keep)
    return "No acute cardiopulmonary abnormality." if (is_normal(report) or report.strip()) else ""


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, classes: list[str], split: str | None, image_size: int = 448,
                 root: str = "", train: bool = False, target_mode: str = "evidence",
                 max_items: int | None = None, require_report: bool = False):
        df = pd.read_csv(manifest, low_memory=False)
        if split is not None:
            df = df[df["split"] == split]
        if require_report:
            df = df[df["report"].fillna("").str.len() > 0]
        if max_items:
            df = df.sample(n=min(max_items, len(df)), random_state=0)
        self.df = df.reset_index(drop=True)
        self.classes, self.size, self.root, self.train = classes, image_size, root, train
        self.target_mode = target_mode
        for c in classes:
            if f"y_{c}" not in self.df.columns:
                self.df[f"y_{c}"] = np.nan

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, p: str) -> str:
        return p if os.path.isabs(p) or not self.root else os.path.join(self.root, p)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        img = _load_gray(self._path(r["image_path"]))
        h0, w0 = img.shape
        img, scale, px, py = letterbox(img, self.size)
        anat = np.zeros((2, self.size, self.size), dtype=np.float32)
        has_anat = False
        if isinstance(r.get("anat_mask", ""), str) and r.get("anat_mask", ""):
            z = np.load(self._path(r["anat_mask"]))
            for k, key in enumerate(["lungs", "heart"]):
                if key in z:
                    anat[k] = letterbox(z[key].astype(np.uint8) * 255, self.size)[0] / 255.0
            has_anat = True
        x = torch.from_numpy(img.copy()).float().div(255).unsqueeze(0)
        m = torch.from_numpy(anat)
        if self.train:
            x, m = self._augment(x, m)
        x = x.repeat(3, 1, 1)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]

        y = torch.tensor([r[f"y_{c}"] if not pd.isna(r[f"y_{c}"]) else -2 for c in self.classes], dtype=torch.float32)
        y_mask = (y > -2).float()
        y = torch.where(y == -1, torch.ones_like(y), y).clamp(min=0)  # U-Ones; -2 (missing) masked anyway

        boxes = []
        if isinstance(r.get("boxes", ""), str) and r.get("boxes", "").strip().startswith("["):
            for b in json.loads(r["boxes"]):
                if b["finding"] in self.classes or b["finding"] == "abnormal":
                    ci = self.classes.index(b["finding"]) if b["finding"] in self.classes else -1
                    boxes.append([ci,
                                  (b["x"] + px) * scale, (b["y"] + py) * scale,
                                  b["w"] * scale, b["h"] * scale])
        report = r.get("report", "")
        report = report if isinstance(report, str) else ""
        target = evidence_sentences(report, self.classes) if self.target_mode == "evidence" else report
        return {"image": x, "y": y, "y_mask": y_mask, "anat": m, "has_anat": float(has_anat),
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5), "text": target,
                "idx": i, "dataset": r.get("dataset", ""), "patient_id": str(r.get("patient_id", i))}

    @staticmethod
    def _augment(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ang = math.radians(random.uniform(-7, 7))
        sc = random.uniform(0.9, 1.1)
        tx, ty = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
        theta = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, tx],
                              [math.sin(ang) / sc, math.cos(ang) / sc, ty]], dtype=torch.float32)[None]
        stack = torch.cat([x, m], 0)[None]
        grid = F.affine_grid(theta, stack.shape, align_corners=False)
        out = F.grid_sample(stack, grid, align_corners=False, padding_mode="zeros")[0]
        xo = out[:1]
        xo = (xo * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        return xo, (out[1:] > 0.5).float()


class Collator:
    """Tokenises explanation targets and builds per-finding token masks (B, C, T)."""

    def __init__(self, tokenizer, classes: list[str], max_len: int = 128):
        self.tok, self.classes, self.max_len = tokenizer, classes, max_len

    def __call__(self, batch: list[dict]) -> dict:
        out = {k: torch.stack([b[k] for b in batch]) for k in ["image", "y", "y_mask", "anat"]}
        out["has_anat"] = torch.tensor([b["has_anat"] for b in batch])
        out["boxes"] = [b["boxes"] for b in batch]
        out["idx"] = torch.tensor([b["idx"] for b in batch])
        out["patient_id"] = [b["patient_id"] for b in batch]
        texts = [b["text"] for b in batch]
        out["texts"] = texts
        if self.tok is None:
            return out
        eos = self.tok.eos_token or ""
        enc = self.tok([t + eos for t in texts], padding="max_length", truncation=True,
                       max_length=self.max_len, return_offsets_mapping=True, return_tensors="pt")
        tmask = torch.zeros(len(batch), len(self.classes), self.max_len)
        for i, t in enumerate(texts):
            spans = positive_sentence_spans(t, self.classes)
            offs = enc["offset_mapping"][i].tolist()
            for c, sp in enumerate(self.classes):
                for (s, e) in spans[sp]:
                    for j, (a, b) in enumerate(offs):
                        if b > a and a >= s and b <= e:
                            tmask[i, c, j] = 1.0
        out["tgt_ids"] = enc["input_ids"]
        out["tgt_attn"] = enc["attention_mask"]
        out["finding_tok_mask"] = tmask * enc["attention_mask"][:, None, :]
        out["has_text"] = torch.tensor([float(len(t) > 0) for t in texts])
        return out


def mentioned_findings(text: str, classes: list[str]) -> list[int]:
    """Binary vector of findings positively asserted in a (generated) text."""
    pos = {m.finding for m in extract_mentions(text) if m.status == 1}
    return [int(c in pos) for c in classes]
