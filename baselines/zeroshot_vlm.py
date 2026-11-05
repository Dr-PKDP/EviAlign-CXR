"""Zero-shot report generation with an off-the-shelf open VLM (no grounding maps), scored with the same
labeler-based clinical metrics. Works with HF image-text-to-text chat models, e.g.
  Qwen/Qwen2.5-VL-7B-Instruct (general domain), google/medgemma-4b-it (gated; check data overlap with NIH).
python baselines/zeroshot_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --sets configs/eval_sets.yaml --out runs/zs_qwen25vl
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from __future__ import annotations

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_items", type=int, default=2000)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    os.makedirs(a.out, exist_ok=True)
    summ = {}
    for s in yaml.safe_load(open(a.sets))["sets"]:
        df = pd.read_csv(s["manifest"], low_memory=False)
        df = df[df["split"] == s.get("split", "test")]
        df = df.sample(n=min(a.max_items, len(df)), random_state=0).reset_index(drop=True)
        gens = []
        for p in df["image_path"]:
            msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")}, {"type": "text", "text": PROMPT}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
            gens.append(proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        df["generated"] = gens
        cls = [c for c in DEFAULT_CLASSES if f"y_{c}" in df.columns and df[f"y_{c}"].notna().any()]
        if cls:
            y = df[[f"y_{c}" for c in cls]].replace(-1, 1).fillna(0).values
            mk = df[[f"y_{c}" for c in cls]].notna().values.astype(float)
            ment = np.array([mentioned_findings(t, cls) for t in gens])
            r = prediction_explanation_consistency(ment, ment, y, mk)   # no classifier: only text-vs-GT terms apply
            summ[s["name"]] = {"hallucination_vs_gt": r["hallucination_vs_gt"], "text_clinical_macro_f1": r["text_clinical_macro_f1"]}
        if "report" in df and df["report"].fillna("").str.len().sum():
            pr = [(x, g) for x, g in zip(df["report"].fillna(""), gens) if x]
            summ.setdefault(s["name"], {}).update(bleu4=corpus_bleu4(*zip(*pr)), rougeL=rouge_l(*zip(*pr)))
        df.to_csv(os.path.join(a.out, f"generated_{s['name']}.csv"), index=False)
        json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
        print(s["name"], summ.get(s["name"]))


if __name__ == "__main__":
    main()
