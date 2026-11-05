"""Word-level tokenizer for CPU smoke tests (mimics the HF fast-tokenizer calls the code uses)."""
from __future__ import annotations

import re

import torch

WORDS = ("findings : there is a are no the right left upper middle lower lung zone lobe base apex "
         "pleural effusion small moderate large consolidation cardiomegaly heart size enlarged "
         "pneumothorax nodule mass opacity atelectasis edema pneumonia emphysema fibrosis hernia "
         "acute cardiopulmonary abnormality clear lungs . , with and of in at").split()


class TinyTokenizer:
    def __init__(self):
        self.special = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.itos = self.special + WORDS
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk = 0, 1, 2, 3
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.vocab_size = len(self.itos)

    def _tok(self, text: str):
        out = []
        for m in re.finditer(r"<eos>|[a-z]+|[.,:]", text.lower()):
            w = m.group(0)
            out.append((self.stoi.get(w, self.unk), m.start(), m.end()))
        return out

    def encode_plain(self, text: str):
        return [t for t, _, _ in self._tok(text)]

    def __call__(self, texts, padding="max_length", truncation=True, max_length=64, return_offsets_mapping=False,
                 return_tensors="pt", add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        ids, am, offs = [], [], []
        for t in texts:
            tk = self._tok(t)[:max_length]
            i = [a for a, _, _ in tk]
            o = [(s, e) for _, s, e in tk]
            pad = max_length - len(i)
            ids.append(i + [self.pad_token_id] * pad)
            am.append([1] * len(i) + [0] * pad)
            offs.append(o + [(0, 0)] * pad)
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor(offs)
        return out

    def batch_decode(self, ids, skip_special_tokens=True):
        res = []
        for row in ids.tolist() if torch.is_tensor(ids) else ids:
            ws = []
            for i in row:
                if i == self.eos_token_id:
                    break
                if skip_special_tokens and i < len(self.special):
                    continue
                ws.append(self.itos[i])
            res.append(" ".join(ws).replace(" .", ".").replace(" ,", ","))
        return res
