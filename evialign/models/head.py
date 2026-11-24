"""Decomposable evidence head.

For class c and head h, a learnable query attends over patch tokens z_m:
    beta^h_{c,m} = softmax_m( q^h_c . W^h_k z_m / sqrt(d) )
and the logit is a sum of per-patch terms
    s_c = sum_h sum_m beta^h_{c,m} * (w^h_c . W^h_v z_m) + b_c  =  sum_m phi_{c,m} + b_c.
phi_{c,m} is therefore an exact additive decomposition of the head output over patches
(completeness holds by construction, as for CAM with a linear GAP head). Positive phi is evidence
for c, negative phi is counter-evidence. Because tokens z_m are contextualised by self-attention,
phi attributes the head output to token positions, not to isolated pixels.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.C, self.H, self.d = n_classes, heads, d
        self.norm = nn.LayerNorm(in_dim)
        self.k = nn.Linear(in_dim, heads * d)
        self.v = nn.Linear(in_dim, heads * d)
        self.q = nn.Parameter(torch.randn(heads, n_classes, d) * d ** -0.5)
        self.w = nn.Parameter(torch.randn(heads, n_classes, d) * d ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_classes))
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = z.shape
        z = self.drop(self.norm(z))
        K = self.k(z).view(B, M, self.H, self.d)
        V = self.v(z).view(B, M, self.H, self.d)
        att = torch.einsum("hcd,bmhd->bhcm", self.q, K) / math.sqrt(self.d)
        beta = att.softmax(-1)                                  # B,H,C,M
        val = torch.einsum("hcd,bmhd->bhcm", self.w, V)         # B,H,C,M
        phi = (beta * val).sum(1)                               # B,C,M
        logits = phi.sum(-1) + self.b
        return logits, phi, beta.mean(1)
