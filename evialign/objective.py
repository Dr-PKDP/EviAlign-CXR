"""Single-forward training objective (DDP-safe: every extra forward pass, including the
sufficiency/comprehensiveness perturbations, happens inside one module call)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .findings import FINDINGS
from .losses import (alignment_loss, anatomical_loss, evidence_distribution, instance_grounding_contrast,
                     masked_bce, pool_distribution, region_prior, sufficiency_comprehensiveness, text_grounding)


class Objective(nn.Module):
    def __init__(self, model, classes: list[str], loss_cfg: dict, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.model, self.classes, self.cfg = model, classes, loss_cfg
        self.regions = [FINDINGS[c]["region"] for c in classes]
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(len(classes)))

    def forward(self, batch: dict, w: dict) -> dict:
        cfg = self.cfg
        x, y, ym = batch["image"], batch["y"], batch["y_mask"]
        use_text = w["lm"] > 0 and "tgt_ids" in batch
        out = self.model(x, batch["tgt_ids"] if use_text else None, batch["tgt_attn"] if use_text else None)
        L = {"cls": masked_bce(out["logits"], y, ym, self.pos_weight if cfg.get("use_pos_weight", True) else None)}
        phi = out["phi"]
        M = phi.shape[-1]
        g = int(math.isqrt(M))
        a_full = evidence_distribution(phi)
        zero = L["cls"] * 0.0

        if use_text:
            has_text = batch["has_text"]
            L["lm"] = (out["lm_loss"] * has_text).sum() / has_text.sum().clamp_min(1)
            p = self.model.mem_grid
            a_mem = pool_distribution(a_full, g, p)
            if w["align"] > 0 or w["inst"] > 0:
                tok = batch["finding_tok_mask"]
                if cfg.get("align_granularity", "finding") == "aggregate":
                    # CAMANet-style control: one class-agnostic evidence map vs. mean attention of all tokens
                    wy = (y * ym)[..., None]
                    a_agg = (a_mem * wy).sum(1, keepdim=True) / wy.sum(1, keepdim=True).clamp_min(1e-8)
                    a_agg = a_agg / a_agg.sum(-1, keepdim=True).clamp_min(1e-8)
                    tok_all = batch["tgt_attn"][:, None, :].float()
                    b, valid = text_grounding(out["grounding"], tok_all)
                    valid = valid & ((y * ym).sum(1, keepdim=True) > 0)
                    a_use = a_agg
                else:
                    b, valid = text_grounding(out["grounding"], tok)
                    valid = valid & (y > 0.5) & (ym > 0)
                    a_use = a_mem
                L["align"] = alignment_loss(a_use, b, valid, cfg.get("align_kind", "js"), cfg.get("align_stop_grad", True))
                L["inst"] = (instance_grounding_contrast(a_use, b, valid, cfg.get("inst_tau", 0.1), cfg.get("align_stop_grad", True))
                             if a_use.shape[1] > 0 else zero)
                L["n_align_pairs"] = valid.sum().float().detach()
        if w["anat"] > 0:
            prior = region_prior(batch["anat"], g, self.regions, cfg.get("anat_dilate", 1))
            L["anat"] = anatomical_loss(a_full, prior, y, ym, batch["has_anat"])
        if w["suff"] > 0 or w["comp"] > 0:
            ls, lc = sufficiency_comprehensiveness(self.model, x, a_full, y * ym, cfg.get("sc_frac", 0.15),
                                                   cfg.get("sc_margin", 1.0), cfg.get("sc_sigma", 8.0))
            L["suff"], L["comp"] = ls, lc
        total = sum(w.get(k, 0.0) * v for k, v in L.items() if k in w)
        L["total"] = total
        L["_logits"] = out["logits"].detach()
        return L
