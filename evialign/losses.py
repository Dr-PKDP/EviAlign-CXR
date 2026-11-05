"""Training objectives.

Notation: phi (B, C, M) exact patch contributions from the evidence head on an encoder grid g x g;
a = evidence distribution (positive part, renormalised); b = text grounding distribution of the
sentences that assert finding c, on the decoder's visual-token grid p x p.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_bce(logits, y, y_mask, pos_weight=None):
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (l * y_mask).sum() / y_mask.sum().clamp_min(1)


def evidence_distribution(phi: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Positive evidence normalised to a distribution, with an epsilon-uniform floor so that
    divergences stay finite. Scale-free (unlike the softmax used in the draft)."""
    p = F.relu(phi)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    M = p.shape[-1]
    return (1 - eps) * p + eps / M


def pool_distribution(d: torch.Tensor, g: int, p: int) -> torch.Tensor:
    """Mass-preserving pooling of a (..., g*g) distribution to (..., p*p)."""
    if g == p:
        return d
    sh = d.shape[:-1]
    x = d.reshape(-1, 1, g, g)
    x = F.adaptive_avg_pool2d(x, p) * (g * g) / (p * p)
    x = x.reshape(*sh, p * p)
    return x / x.sum(-1, keepdim=True).clamp_min(1e-8)


def text_grounding(att: torch.Tensor, tok_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """att (B,T,P), tok_mask (B,C,T) -> b (B,C,P) and validity (B,C)."""
    w = tok_mask / tok_mask.sum(-1, keepdim=True).clamp_min(1e-8)
    b = torch.einsum("bct,btp->bcp", w, att)
    valid = tok_mask.sum(-1) > 0
    P = att.shape[-1]
    b = (1 - 1e-3) * b / b.sum(-1, keepdim=True).clamp_min(1e-8) + 1e-3 / P
    return b, valid


def js_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)


def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p / q).log()).sum(-1)


def alignment_loss(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor, kind: str = "js",
                   stop_grad: bool = True) -> torch.Tensor:
    """Cross-modal evidence agreement for (image, finding) pairs where the reference text asserts
    the finding and the label is positive. a: (B,C,P) classifier evidence; b: (B,C,P) text grounding.
    kind: 'js' (default, symmetric, bounded), 'kl_ba' = KL(b||a) penalises text mass on patches
    without classifier evidence, 'kl_ab' = KL(a||b) penalises evidence the text ignores."""
    if valid.sum() == 0:
        return a.sum() * 0.0
    a = a.detach() if stop_grad else a
    if kind == "js":
        d = js_div(a, b)
    elif kind == "kl_ba":
        d = kl_div(b, a)
    elif kind == "kl_ab":
        d = kl_div(a, b)
    else:
        raise ValueError(kind)
    return d[valid].mean()


def instance_grounding_contrast(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor, tau: float = 0.1,
                                stop_grad: bool = True) -> torch.Tensor:
    """For each finding c, the text grounding of image i must match the evidence map of image i
    better than the evidence maps of other images j that carry the same finding (InfoNCE over the
    batch). This blocks the shortcut of grounding a finding to its dataset-average location."""
    a = a.detach() if stop_grad else a
    losses = []
    for c in range(a.shape[1]):
        idx = valid[:, c].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        A = F.normalize(a[idx, c] - a[idx, c].mean(-1, keepdim=True), dim=-1)
        Bm = F.normalize(b[idx, c] - b[idx, c].mean(-1, keepdim=True), dim=-1)
        S = Bm @ A.t() / tau
        tgt = torch.arange(idx.numel(), device=a.device)
        losses.append(F.cross_entropy(S, tgt))
    if not losses:
        return a.sum() * 0.0
    return torch.stack(losses).mean()


def region_prior(anat: torch.Tensor, g: int, regions: list[str], dilate: int = 1) -> torch.Tensor:
    """Binary per-class plausibility masks (B, C, g*g) from lung/heart masks (B, 2, H, W)."""
    lung = F.adaptive_max_pool2d(anat[:, :1], g)
    heart = F.adaptive_max_pool2d(anat[:, 1:2], g)
    if dilate > 0:
        k = 2 * dilate + 1
        lung = F.max_pool2d(lung, k, 1, dilate)
        heart = F.max_pool2d(heart, k, 1, dilate)
    lung, heart = lung.flatten(1), heart.flatten(1)
    both = torch.maximum(lung, heart)
    cols = []
    for r in regions:
        cols.append({"lungs": both, "heart": heart, "any": torch.ones_like(lung)}.get(r, torch.ones_like(lung)))
    return torch.stack(cols, 1)


def anatomical_loss(a: torch.Tensor, prior: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor,
                    has_anat: torch.Tensor) -> torch.Tensor:
    """Evidence mass that a positive finding places outside its anatomically plausible region.
    Gradients flow into the classifier (a is not detached): 'right for the right reasons' with weak,
    automatically derived anatomy masks instead of lesion annotations."""
    outside = (a * (1 - prior)).sum(-1)                         # B, C
    w = y * y_mask * has_anat[:, None]
    return (outside * w).sum() / w.sum().clamp_min(1)


def soft_topk_mask(a: torch.Tensor, frac: float, temp: float = 0.1) -> torch.Tensor:
    """Differentiable keep-mask over patches that retains roughly the top ``frac`` of evidence."""
    k = max(1, int(round(frac * a.shape[-1])))
    thr = a.detach().topk(k, dim=-1).values[..., -1:]
    scale = a.detach().std(-1, keepdim=True).clamp_min(1e-6) * temp
    return torch.sigmoid((a - thr) / scale)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(2 * math.ceil(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-t ** 2 / (2 * sigma ** 2))
    w = (w / w.sum()).view(1, 1, 1, k)
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.expand(C, 1, 1, k), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.transpose(2, 3).expand(C, 1, k, 1), groups=C)
    return x


def perturb(x: torch.Tensor, keep: torch.Tensor, g: int, blurred: torch.Tensor) -> torch.Tensor:
    """keep: (B, g*g) in [0,1]; regions with keep=0 are replaced by a blurred copy (an in-distribution
    'removal' baseline, avoiding the black-patch artefacts that ROAR warns about)."""
    m = keep.view(-1, 1, g, g)
    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return m * x + (1 - m) * blurred


def sufficiency_comprehensiveness(model, x, a_full, y, frac: float, margin: float, sigma: float):
    """Corrected sufficiency / comprehensiveness regulariser on one positive class per image.
       L_suff = relu( s_c(x) - s_c(x_keep) )             keeping the evidence must not lower the score
       L_comp = relu( margin - (s_c(x) - s_c(x_remove)) ) removing it must lower the score by >= margin
    s_c(x) is detached so that neither term can be met by lowering the clean score. (The draft
    minimised |s_c(x_remove) - s_c(x)|, which rewards the opposite of comprehensiveness.)"""
    B, C, M = a_full.shape
    g = int(math.isqrt(M))
    pos = y > 0.5
    has = pos.any(1)
    if has.sum() == 0:
        return x.sum() * 0.0, x.sum() * 0.0
    r = torch.rand(B, C, device=x.device) * pos
    c = r.argmax(1)
    bi = torch.arange(B, device=x.device)
    a_c = a_full[bi, c]
    keep = soft_topk_mask(a_c, frac)
    blurred = gaussian_blur(x, sigma)
    with torch.no_grad():
        s_clean = model.classify(model.encode(x))[0][bi, c]
    s_keep = model.classify(model.encode(perturb(x, keep, g, blurred)))[0][bi, c]
    s_rm = model.classify(model.encode(perturb(x, 1 - keep, g, blurred)))[0][bi, c]
    l_suff = F.relu(s_clean - s_keep)[has].mean()
    l_comp = F.relu(margin - (s_clean - s_rm))[has].mean()
    return l_suff, l_comp
