"""Frontal-vs-lateral view filter by BiomedCLIP zero-shot (used when a corpus lacks view metadata).
Validate on a few hundred images by eye; the threshold can be tuned with --thr."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

PROMPTS = ["a frontal chest x-ray", "a posteroanterior chest radiograph", "an anteroposterior chest radiograph",
           "a lateral chest x-ray", "a lateral chest radiograph"]
FRONTAL = [0, 1, 2]


@torch.no_grad()
def frontal_probability(paths: list[str], bs: int = 128) -> np.ndarray:
    import open_clip
    name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, preprocess = open_clip.create_model_from_pretrained(name)
    tok = open_clip.get_tokenizer(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    t = model.encode_text(tok(PROMPTS).to(dev))
    t = t / t.norm(dim=-1, keepdim=True)
    out = []
    for i in range(0, len(paths), bs):
        ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        f = model.encode_image(ims)
        f = f / f.norm(dim=-1, keepdim=True)
        pr = (100 * f @ t.T).softmax(-1)
        out.append(pr[:, FRONTAL].sum(-1).cpu().numpy())
    return np.concatenate(out)


class _Imgs(torch.utils.data.Dataset):
    def __init__(self, paths, pre):
        self.p, self.pre = paths, pre

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.pre(Image.open(self.p[i]).convert("RGB"))


@torch.no_grad()
def embed_images(paths: list[str], bs: int = 64, workers: int = 8) -> np.ndarray:
    """L2-normalised BiomedCLIP image embeddings."""
    import open_clip
    name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, pre = open_clip.create_model_from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    dl = torch.utils.data.DataLoader(_Imgs(paths, pre), batch_size=bs, num_workers=workers)
    out = []
    for x in dl:
        f = model.encode_image(x.to(dev))
        out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out)


def view_probe_frontal_probability(frontal_ref: list[str], lateral_ref: list[str], query: list[str], seed: int = 0):
    """Linear probe on BiomedCLIP embeddings trained on images with known view labels; returns p(frontal) for the
    query images, the held-out accuracy/AUROC of the probe and the fitted classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    paths = list(frontal_ref) + list(lateral_ref) + list(query)
    E = embed_images(paths)
    nf, nl = len(frontal_ref), len(lateral_ref)
    X, y = E[:nf + nl], np.r_[np.ones(nf), np.zeros(nl)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
    pte = clf.predict_proba(X[te])[:, 1]
    stats = {"heldout_acc": float(((pte >= .5) == (y[te] == 1)).mean()), "heldout_auroc": float(roc_auc_score(y[te], pte)),
             "n_train": int(len(tr)), "n_test": int(len(te))}
    clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(X, y)
    return clf.predict_proba(E[nf + nl:])[:, 1], stats, clf


def apply_view_probe(clf_path: str, paths: list[str]) -> np.ndarray:
    import joblib
    return joblib.load(clf_path).predict_proba(embed_images(paths))[:, 1]


def adapt_view_probe(clf_path: str, paths: list[str], groups: list, n_folds: int = 5, seed: int = 0):
    """Domain adaptation of a saved view probe to a corpus without view labels. Within every group (report) that has two or
    more images, the image the source probe scores highest is a pseudo-frontal example and the lowest a pseudo-lateral one.
    A new probe is fitted on the corpus' own embeddings, cross-fitted by group so that no image is scored by a model that
    saw its report. Returns p(frontal) for every image, the source-probe scores and agreement statistics."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    E = embed_images(paths)
    p1 = joblib.load(clf_path).predict_proba(E)[:, 1]
    groups = np.asarray([str(g) for g in groups])
    y = np.full(len(paths), -1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) >= 2:
            y[idx[np.argmax(p1[idx])]] = 1
            y[idx[np.argmin(p1[idx])]] = 0
    p2 = np.zeros(len(paths))
    for tr, te in GroupKFold(n_splits=n_folds).split(E, groups=groups):
        lab = tr[y[tr] >= 0]
        clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(E[lab], y[lab])
        p2[te] = clf.predict_proba(E[te])[:, 1]
    m = y >= 0
    stats = {"pseudo_labelled": int(m.sum()), "cv_agreement_with_pseudo_labels": float(((p2[m] >= .5) == (y[m] == 1)).mean())}
    return p2, p1, stats
