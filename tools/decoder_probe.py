"""Three checks on what the decoder does with the image.

1. Absolute attention mass. The grounding of Section 3.3 renormalises attention over the visual prefix,
   which hides how much attention a token gives the image at all. This records the mass before
   renormalisation, for the tokens that assert a finding and for the remaining target tokens.
2. Image-substitution control. The same target text is scored against the visual tokens of another image.
   If the per-token log-likelihood does not fall and the grounding does not move, the wording does not
   depend on the image.
3. Zero positive evidence. How often a (image, finding) pair has no positive patch contribution, the case
   in which the evidence distribution falls back to the epsilon floor.

python tools/decoder_probe.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 512 --out results/decoder_probe.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from evialign.data import Collator, ManifestDataset
from evialign.losses import evidence_distribution, js_div, pool_distribution, text_grounding
from evialign.models.model import EviAlign
from evaluate import token_masks


def prefix_attention(dec, memory, ids, attn):
    """Re-runs PrefixLLMDecoder._run and returns (renormalised grounding, absolute prefix mass, token logprobs)."""
    pre, P = dec._prefix(memory)
    L0 = pre.shape[1]
    emb = torch.cat([pre, dec._emb(ids)], 1)
    am = torch.cat([torch.ones(ids.shape[0], L0, dtype=attn.dtype, device=attn.device), attn], 1)
    out = dec.lm(inputs_embeds=emb, attention_mask=am, output_attentions=True, return_dict=True)
    T = ids.shape[1]
    rows = slice(L0 - 1, L0 - 1 + T)
    att = torch.stack([a[:, :, rows, :P] for a in out.attentions[-dec.ng:]], 0).mean((0, 2)).float()
    mass = att.sum(-1)                                                   # B,T  attention on the image
    g = att / att.sum(-1, keepdim=True).clamp_min(1e-8)
    lp = out.logits[:, rows, :].float().log_softmax(-1).gather(-1, ids[..., None])[..., 0]
    return g, mass, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out", default="results/decoder_probe.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    classes = cfg["data"]["classes"]
    model = EviAlign(cfg, len(classes))
    model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    tok = model.decoder.tok
    ds = ManifestDataset(a.manifest, classes, a.split, cfg["data"]["image_size"], max_items=a.n)
    dl = DataLoader(ds, batch_size=a.batch_size, collate_fn=Collator(None, classes))

    mass_find, mass_other, lp_true, lp_shuf, js_shuf, js_self = [], [], [], [], [], []
    zero_pos, n_pairs, n_pos_pairs = 0, 0, 0
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev)
            z = model.encode(x)
            logits, phi, _ = model.classify(z)
            pos = torch.relu(phi).sum(-1)                                # B,C positive evidence mass
            zero_pos += int((pos <= 0).sum()); n_pairs += pos.numel()
            mem = model.memory(z)
            gen = model.decoder.generate_ids(mem, max_new_tokens=a.max_new_tokens)
            texts = [tok.decode(i, skip_special_tokens=True) for i in gen[0]]
            ids, attn, tm = token_masks(tok, texts, classes, gen[0].shape[1])
            ids, attn, tm = ids.to(dev), attn.to(dev), tm.to(dev)
            g, mass, lp = prefix_attention(model.decoder, mem, ids, attn)
            find_tok = (tm.sum(1) > 0).float() * attn                    # tokens inside an asserting sentence
            other_tok = (1 - (tm.sum(1) > 0).float()) * attn
            mass_find += mass[find_tok > 0].tolist()
            mass_other += mass[other_tok > 0].tolist()
            lp_true += ((lp * attn).sum(1) / attn.sum(1).clamp_min(1)).tolist()
            # the same text against another image's visual tokens
            g2, _, lp2 = prefix_attention(model.decoder, mem.roll(1, 0), ids, attn)
            lp_shuf += ((lp2 * attn).sum(1) / attn.sum(1).clamp_min(1)).tolist()
            bb, valid = text_grounding(g, tm)
            b2, _ = text_grounding(g2, tm)
            aa = pool_distribution(evidence_distribution(phi), int(round(phi.shape[-1] ** 0.5)), int(round(bb.shape[-1] ** 0.5)))
            v = valid & (logits.sigmoid() > 0.5)
            if v.any():
                js_self += js_div(aa[v], bb[v]).tolist()                 # grounding vs own evidence
                js_shuf += js_div(bb[v], b2[v]).tolist()                 # grounding with the image swapped
                n_pos_pairs += int(v.sum())
    q = lambda v: [float(np.percentile(v, p)) for p in (25, 50, 75)] if v else []
    res = {"n_images": len(ds), "pairs": n_pairs, "zero_positive_evidence": zero_pos,
           "zero_positive_evidence_share": zero_pos / max(1, n_pairs),
           "prefix_mass_finding_tokens": {"mean": float(np.mean(mass_find)), "quartiles": q(mass_find)},
           "prefix_mass_other_tokens": {"mean": float(np.mean(mass_other)), "quartiles": q(mass_other)},
           "logprob_per_token": {"own_image": float(np.mean(lp_true)), "other_image": float(np.mean(lp_shuf)),
                                 "mean_drop": float(np.mean(lp_true) - np.mean(lp_shuf))},
           "js_grounding_vs_evidence": {"mean": float(np.mean(js_self)), "quartiles": q(js_self)},
           "js_grounding_own_vs_other_image": {"mean": float(np.mean(js_shuf)), "quartiles": q(js_shuf)},
           "n_scored_pairs": n_pos_pairs}
    _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
