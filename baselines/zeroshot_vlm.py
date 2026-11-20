"""Zero-shot report generation with an off-the-shelf open VLM (no grounding maps), scored with the same
labeler-based clinical metrics as the trained systems (hallucination against the labels, text clinical F1, BLEU-4 and ROUGE-L).
Works with HF image-text-to-text chat models, e.g. Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0, general domain).
Each set is a fixed random subset (--max_items, seed 0, drawn after the split filter exactly as in evaluate.py).

python baselines/zeroshot_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --sets configs/eval_zs.yaml --out runs/zs_qwen25vl
"""
from __future__ import annotations

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from evialign.data import mentioned_findings
from evialign.findings import DEFAULT_CLASSES
from evialign.metrics import corpus_bleu4, prediction_explanation_consistency, rouge_l

PROMPT = "You are a radiologist. Describe the findings of this chest X-ray in a few sentences."


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("I;16", "I"):   # 16-bit PNG: scale to 8 bit before the RGB conversion
        a = np.asarray(im).astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        im = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8))
    return im.convert("RGB")


def generate(proc, model, images, max_new_tokens, device):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}] for im in images]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return [t.strip() for t in proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_items", type=int, default=2000)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map=device).eval()
    os.makedirs(a.out, exist_ok=True)
    summ = {}
    for s in yaml.safe_load(open(a.sets))["sets"]:
        if not os.path.exists(s["manifest"]):
            print("skip (no manifest)", s["name"])
            continue
        df = pd.read_csv(s["manifest"], low_memory=False)
        df = df[df["split"] == s.get("split", "test")]
        df = df.sample(n=min(a.max_items, len(df)), random_state=0).reset_index(drop=True)
        root = s.get("root", "")
        paths = [p if os.path.isabs(p) or not root else os.path.join(root, p) for p in df["image_path"]]
        gens = []
        for i in range(0, len(paths), a.batch_size):
            ims = [load_image(p) for p in paths[i:i + a.batch_size]]
            try:
                gens += generate(proc, model, ims, a.max_new_tokens, device)
            except Exception as e:   # fall back to single-image generation for this batch
                print("batch failed, retrying singly:", str(e)[:120], flush=True)
                for im in ims:
                    gens += generate(proc, model, [im], a.max_new_tokens, device)
            if (i // a.batch_size) % 25 == 0:
                print(s["name"], len(gens), "/", len(paths), flush=True)
        df["generated"] = gens
        cls = [c for c in DEFAULT_CLASSES if f"y_{c}" in df.columns and df[f"y_{c}"].notna().any()]
        if cls:
            y = df[[f"y_{c}" for c in cls]].replace(-1, 1).fillna(0).values
            mk = df[[f"y_{c}" for c in cls]].notna().values.astype(float)
            ment = np.array([mentioned_findings(t, cls) for t in gens])
            r = prediction_explanation_consistency(ment, ment, y, mk)   # no classifier: only the text-versus-labels terms apply
            summ[s["name"]] = {"n": len(df), "hallucination_vs_gt": r["hallucination_vs_gt"],
                               "text_clinical_macro_f1": r["text_clinical_macro_f1"],
                               "share_normal_statement": float(np.mean([not m.any() for m in ment]))}
        if "report" in df and df["report"].fillna("").str.len().sum():
            pr = [(x, g) for x, g in zip(df["report"].fillna(""), gens) if x]
            summ.setdefault(s["name"], {}).update(bleu4=corpus_bleu4(*zip(*pr)), rougeL=rouge_l(*zip(*pr)))
        df.to_csv(os.path.join(a.out, f"generated_{s['name']}.csv"), index=False)
        json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
        print(s["name"], summ.get(s["name"]), flush=True)


if __name__ == "__main__":
    main()
