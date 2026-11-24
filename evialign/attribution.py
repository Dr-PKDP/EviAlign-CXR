"""Post-hoc attribution baselines computed on the same model, returned on the encoder patch grid
(B, M) so that every map is evaluated with identical metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def evidence_map(model, x, c):
    """Intrinsic exact evidence phi_c (positive part)."""
    with torch.no_grad():
        logits, phi, _ = model.classify(model.encode(x))
    return F.relu(phi[torch.arange(x.shape[0]), c])


def attention_map(model, x, c):
    with torch.no_grad():
        _, _, beta = model.classify(model.encode(x))
    return beta[torch.arange(x.shape[0]), c]


def gradcam_tokens(model, x, c):
    """Grad-CAM adapted to ViT patch tokens: weights = token-averaged gradient of s_c."""
    with torch.enable_grad():
        z = model.encode(x).detach().requires_grad_(True)
        logits, _, _ = model.classify(z)
        s = logits[torch.arange(x.shape[0]), c].sum()
        (g,) = torch.autograd.grad(s, z)
    w = g.mean(1, keepdim=True)                         # B,1,D
    return F.relu((w * z.detach()).sum(-1))             # B,M


def integrated_gradients(model, x, c, steps: int = 32, baseline: torch.Tensor | None = None):
    """Pixel IG (blurred-image baseline by default), aggregated to patch-grid |IG| mass."""
    from .losses import gaussian_blur
    base = gaussian_blur(x, sigma=x.shape[-1] / 40) if baseline is None else baseline
    total = torch.zeros_like(x)
    bi = torch.arange(x.shape[0])
    with torch.enable_grad():
        for k in range(1, steps + 1):
            xi = (base + (k / steps) * (x - base)).detach().requires_grad_(True)
            s = model.classify(model.encode(xi))[0][bi, c].sum()
            (g,) = torch.autograd.grad(s, xi)
            total += g
    ig = ((x - base) * total / steps).abs().sum(1, keepdim=True)
    gsz = model.encoder.grid
    return F.adaptive_avg_pool2d(ig, gsz).flatten(1)


METHODS = {"evidence": evidence_map, "attention": attention_map, "gradcam": gradcam_tokens, "ig": integrated_gradients}
