"""CheXbert report labeler (Smit et al., EMNLP 2020), re-implemented for inference only.

Checkpoint: huggingface.co/StanfordAIMI/RRG_scorers  file chexbert.pth (not gated).
Architecture: bert-base-uncased CLS embedding -> 13 heads x 4 classes (blank/positive/negative/uncertain)
+ 1 head x 2 classes for No Finding.

python -m evialign.prepare.chexbert --in reports.csv --text_col report --out labels.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices", "No Finding"]
TO_CANON = {"Cardiomegaly": "cardiomegaly", "Lung Opacity": "lung_opacity", "Lung Lesion": "nodule_mass", "Edema": "edema",
            "Consolidation": "consolidation", "Pneumonia": "pneumonia", "Atelectasis": "atelectasis",
            "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion", "Fracture": "fracture",
            "No Finding": "no_finding"}


class CheXbert(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(768, 4) for _ in range(13)] + [nn.Linear(768, 2)])

    def forward(self, ids, am):
        h = self.bert(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        return [head(h) for head in self.linear_heads]


def load(device):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = CheXbert()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.endswith("position_ids")], f"missing keys: {missing[:5]}"
    return m.to(device).eval()


@torch.no_grad()
def label_texts(texts: list[str], device="cuda", bs: int = 64) -> pd.DataFrame:
    """Returns canonical columns y_<finding> in {1, 0, -1}; blank -> 0 (not mentioned)."""
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    m = load(device)
    rows = []
    for i in range(0, len(texts), bs):
        chunk = [t if isinstance(t, str) and t.strip() else "" for t in texts[i:i + bs]]
        enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outs = m(enc["input_ids"], enc["attention_mask"])
        cls = torch.stack([o.argmax(-1) for o in outs], 1).cpu().numpy()   # B,14
        rows.append(cls)
    cls = np.concatenate(rows)
    out = {}
    for j, cond in enumerate(CONDITIONS):
        if cond not in TO_CANON:
            continue
        v = cls[:, j]
        if cond == "No Finding":
            out[f"y_{TO_CANON[cond]}"] = (v == 1).astype(int)
        else:   # 0 blank, 1 positive, 2 negative, 3 uncertain
            out[f"y_{TO_CANON[cond]}"] = np.select([v == 1, v == 3], [1, -1], default=0)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--text_col", default="report")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    lab = label_texts(df[a.text_col].fillna("").tolist(), "cuda" if torch.cuda.is_available() else "cpu")
    pd.concat([df.reset_index(drop=True), lab], axis=1).to_csv(a.out, index=False)
