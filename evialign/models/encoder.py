"""Vision encoders that expose a spatial grid of patch tokens.

Backbone strings:
  timm:<name>      e.g. timm:vit_base_patch14_dinov2.lvd142m (general-domain, no CXR exposure)
  hf:<repo>        e.g. hf:microsoft/rad-dino (CXR-specific DINOv2; check its pretraining data
                   against the external test sets before use, see README)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(self, backbone: str, image_size: int, pretrained: bool = True, grad_ckpt: bool = False,
                 timm_kwargs: dict | None = None):
        super().__init__()
        self.kind, name = backbone.split(":", 1)
        if self.kind == "timm":
            import timm
            kw = dict(pretrained=pretrained, num_classes=0, img_size=image_size)
            kw.update(timm_kwargs or {})
            self.net = timm.create_model(name, **kw)
            if grad_ckpt and hasattr(self.net, "set_grad_checkpointing"):
                self.net.set_grad_checkpointing(True)
            self.dim = self.net.num_features
            self.n_prefix = getattr(self.net, "num_prefix_tokens", 1)
            ps = self.net.patch_embed.patch_size
            self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps
        elif self.kind == "openclip":
            # e.g. openclip:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
            import open_clip
            clip, _ = open_clip.create_model_from_pretrained(name)
            self.net = clip.visual.trunk
            self.kind = "timm"
            if hasattr(self.net, "set_input_size"):
                self.net.set_input_size(img_size=image_size)
            if grad_ckpt and hasattr(self.net, "set_grad_checkpointing"):
                self.net.set_grad_checkpointing(True)
            self.dim = self.net.num_features
            self.n_prefix = getattr(self.net, "num_prefix_tokens", 1)
            ps = self.net.patch_embed.patch_size
            self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps
        elif self.kind == "hf":
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(name)
            if grad_ckpt:
                self.net.gradient_checkpointing_enable()
            cfg = self.net.config
            self.dim = cfg.hidden_size
            self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
            self.patch = cfg.patch_size
        else:
            raise ValueError(backbone)
        self.image_size = image_size
        self.grid = image_size // self.patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch tokens (B, M, D) with M = grid * grid (row-major)."""
        if self.kind == "timm":
            t = self.net.forward_features(x)
        else:
            t = self.net(pixel_values=x).last_hidden_state
        z = t[:, self.n_prefix:, :]
        m = z.shape[1]
        g = int(math.isqrt(m))
        assert g * g == m, f"non-square token grid ({m})"
        self.grid = g
        return z
