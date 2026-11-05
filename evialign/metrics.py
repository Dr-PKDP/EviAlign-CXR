"""Evaluation metrics: classification, calibration, localisation, text, and faithfulness."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


# ----------------------------------------------------------------------------- classification
def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def _ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else np.nan


def patient_bootstrap(groups: list[str], n_boot: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        yield np.concatenate([by[keys[k]] for k in pick])


def classification_report(y, mask, prob, classes, groups, n_boot=1000, thresholds=None):
    out = {"per_class": {}, "macro": {}}
    aucs, aps = [], []
    for c, name in enumerate(classes):
        m = mask[:, c] > 0
        if m.sum() == 0 or y[m, c].sum() == 0:
            continue
        yy, pp = y[m, c], prob[m, c]
        r = {"n": int(m.sum()), "n_pos": int(yy.sum()), "auroc": _auc(yy, pp), "auprc": _ap(yy, pp)}
        if thresholds is not None:
            r["f1"] = f1_score(yy, pp >= thresholds[c], zero_division=0)
            r["sens"] = float(((pp >= thresholds[c]) & (yy == 1)).sum() / max(yy.sum(), 1))
            r["spec"] = float(((pp < thresholds[c]) & (yy == 0)).sum() / max((yy == 0).sum(), 1))
        out["per_class"][name] = r
        aucs.append(r["auroc"])
        aps.append(r["auprc"])
    out["macro"]["auroc"] = float(np.nanmean(aucs)) if aucs else np.nan
    out["macro"]["auprc"] = float(np.nanmean(aps)) if aps else np.nan
    if n_boot:
        valid = [c for c, n in enumerate(classes) if n in out["per_class"]]
        boots = defaultdict(list)
        for idx in patient_bootstrap(groups, n_boot):
            per = []
            for c in valid:
                m = mask[idx, c] > 0
                a = _auc(y[idx][m, c], prob[idx][m, c])
                boots[classes[c]].append(a)
                per.append(a)
            boots["macro"].append(np.nanmean(per))
        for k, v in boots.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            (out["macro"] if k == "macro" else out["per_class"][k])["auroc_ci"] = [float(lo), float(hi)]
    return out


def f1_thresholds(y, mask, prob):
    th = np.full(y.shape[1], 0.5)
    for c in range(y.shape[1]):
        m = mask[:, c] > 0
        if m.sum() == 0 or y[m, c].sum() == 0:
            continue
        cand = np.unique(np.quantile(prob[m, c], np.linspace(0.01, 0.99, 199)))
        f = [f1_score(y[m, c], prob[m, c] >= t, zero_division=0) for t in cand]
        th[c] = float(cand[int(np.argmax(f))])
    return th


def ece(y, mask, prob, bins=15):
    m = mask.flatten() > 0
    yy, pp = y.flatten()[m], prob.flatten()[m]
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        s = (pp >= edges[i]) & (pp < edges[i + 1] if i < bins - 1 else pp <= 1)
        if s.sum():
            e += s.mean() * abs(yy[s].mean() - pp[s].mean())
    return float(e)


# ----------------------------------------------------------------------------- localisation
def upsample_map(m: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    g = int(math.isqrt(m.size))
    t = torch.from_numpy(m.reshape(1, 1, g, g).astype(np.float32))
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def box_mask(boxes: np.ndarray, size: int) -> np.ndarray:
    mk = np.zeros((size, size), dtype=bool)
    for (_, x, y, w, h) in boxes:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        mk[y0:min(int(math.ceil(y + h)), size), x0:min(int(math.ceil(x + w)), size)] = True
    return mk


def localisation_scores(sal: np.ndarray, boxes: np.ndarray, size: int, tol_px: int = 15) -> dict:
    """Pointing game (with tolerance, as in Zhang et al.), evidence mass inside boxes (energy-based
    pointing game) and the mass-to-area ratio (energy / box-area fraction; 1 = chance)."""
    s = np.clip(upsample_map(sal, size), 0, None)
    mk = box_mask(boxes, size)
    if mk.sum() == 0 or s.sum() <= 0:
        return {}
    yx = np.unravel_index(np.argmax(s), s.shape)
    y0, x0 = yx
    win = mk[max(0, y0 - tol_px):y0 + tol_px + 1, max(0, x0 - tol_px):x0 + tol_px + 1]
    energy = float(s[mk].sum() / s.sum())
    return {"hit": float(win.any()), "energy": energy, "mass_area_ratio": energy / float(mk.mean())}


# ----------------------------------------------------------------------------- text
def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu4(refs: list[str], hyps: list[str]) -> float:
    pn, pd, hl, rl = [0] * 4, [0] * 4, 0, 0
    for r, h in zip(refs, hyps):
        rt, ht = r.lower().split(), h.lower().split()
        hl += len(ht)
        rl += len(rt)
        for n in range(1, 5):
            hn, rn = _ngrams(ht, n), _ngrams(rt, n)
            pn[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            pd[n - 1] += max(sum(hn.values()), 0)
    if min(pd) == 0 or min(pn) == 0:
        return 0.0
    logp = sum(math.log(pn[i] / pd[i]) for i in range(4)) / 4
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return float(bp * math.exp(logp))


def rouge_l(refs: list[str], hyps: list[str]) -> float:
    def lcs(a, b):
        dp = [0] * (len(b) + 1)
        for x in a:
            prev = 0
            for j, yb in enumerate(b, 1):
                cur = dp[j]
                dp[j] = prev + 1 if x == yb else max(dp[j], dp[j - 1])
                prev = cur
        return dp[-1]
    sc = []
    for r, h in zip(refs, hyps):
        rt, ht = r.lower().split(), h.lower().split()
        if not rt or not ht:
            sc.append(0.0)
            continue
        l = lcs(rt, ht)
        p, rc = l / len(ht), l / len(rt)
        sc.append(0.0 if l == 0 else 2 * p * rc / (p + rc))
    return float(np.mean(sc)) if sc else 0.0


# ----------------------------------------------------------------------------- faithfulness
def prediction_explanation_consistency(pred: np.ndarray, mentioned: np.ndarray, y: np.ndarray,
                                       mask: np.ndarray) -> dict:
    """pred, mentioned, y: (N, C) binary. Consistency between what the classifier predicts and what
    the explanation asserts, and between what the explanation asserts and the reference labels."""
    m = mask > 0
    agree = (pred == mentioned) & m
    ment = mentioned.astype(bool) & m
    prd = pred.astype(bool) & m
    out = {
        "pec_accuracy": float(agree.sum() / max(m.sum(), 1)),
        "pec_macro_f1": float(np.nanmean([f1_score(pred[m[:, c], c], mentioned[m[:, c], c], zero_division=0)
                                          for c in range(pred.shape[1]) if m[:, c].any() and
                                          (pred[m[:, c], c].sum() + mentioned[m[:, c], c].sum()) > 0])),
        # asserted findings the classifier does not predict
        "unsupported_by_classifier": float((ment & ~prd).sum() / max(ment.sum(), 1)),
        # predicted findings the explanation omits
        "omitted_by_text": float((prd & ~ment).sum() / max(prd.sum(), 1)),
        # asserted findings absent from the reference labels (hallucination w.r.t. ground truth)
        "hallucination_vs_gt": float((ment & (y < 0.5)).sum() / max(ment.sum(), 1)),
        "text_clinical_macro_f1": float(np.nanmean([f1_score(y[m[:, c], c], mentioned[m[:, c], c], zero_division=0)
                                                     for c in range(y.shape[1]) if m[:, c].any() and y[m[:, c], c].sum() > 0])),
    }
    return out


def deletion_insertion_auc(score_fn, x, sal, steps: int = 10):
    """Area under the deletion / insertion curves (Petsiuk et al.) using the blurred baseline.
    score_fn(x_batch) -> class score in probability space (B,). sal: (B, M)."""
    import torch
    from .losses import gaussian_blur, perturb
    g = int(math.isqrt(sal.shape[1]))
    blurred = gaussian_blur(x, sigma=x.shape[-1] / 40)
    order = sal.argsort(1, descending=True)
    dele, ins = [], []
    for k in range(steps + 1):
        n = int(round(k / steps * sal.shape[1]))
        keep = torch.ones_like(sal)
        if n:
            keep.scatter_(1, order[:, :n], 0.0)
        dele.append(score_fn(perturb(x, keep, g, blurred)))
        ins.append(score_fn(perturb(x, 1 - keep, g, blurred)))
    d = torch.stack(dele, 1)
    i = torch.stack(ins, 1)
    trap = lambda v: ((v[:, 1:] + v[:, :-1]) / 2).mean(1)
    return trap(d), trap(i)
