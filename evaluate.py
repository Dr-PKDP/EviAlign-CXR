"""Evaluate a checkpoint on internal and external sets.

python evaluate.py --ckpt runs/main/best.pt --sets configs/eval_sets.yaml --out runs/main/eval
Writes, per set: summary.json, per_image.csv, and maps_<set>.npz (a sample of maps for figures).
Thresholds are fixed on the internal validation split and transferred unchanged to external sets.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import resource
from torch.utils.data import DataLoader

try:   # many sets x many workers exhaust the default open-file limit
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ValueError, OSError):
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

from evialign.attribution import gradcam_tokens, integrated_gradients
from evialign.data import Collator, ManifestDataset, mentioned_findings
from evialign.labeler import positive_sentence_spans
from evialign.losses import (evidence_distribution, gaussian_blur, js_div, perturb, pool_distribution,
                             text_grounding)
from evialign.metrics import (classification_report, corpus_bleu4, deletion_insertion_auc, ece, f1_thresholds,
                              localisation_scores, prediction_explanation_consistency, rouge_l)
from evialign.models.model import EviAlign
from evialign.findings import DISPLAY, FINDINGS


def token_masks(tok, texts, classes, max_len):
    enc = tok([t + (tok.eos_token or "") for t in texts], padding="max_length", truncation=True, max_length=max_len,
              return_offsets_mapping=True, return_tensors="pt")
    tm = torch.zeros(len(texts), len(classes), max_len)
    for i, t in enumerate(texts):
        sp = positive_sentence_spans(t, classes)
        for c, name in enumerate(classes):
            for (s, e) in sp[name]:
                for j, (a, b) in enumerate(enc["offset_mapping"][i].tolist()):
                    if b > a and a >= s and b <= e:
                        tm[i, c, j] = 1
    return enc["input_ids"], enc["attention_mask"], tm * enc["attention_mask"][:, None, :]


def opening_masks(tok, texts, classes, tm, max_len):
    """Splits each finding's token mask into the forced canonical opening and the decoder's continuation.

    In finding-conditioned decoding the sentence asserting c starts with FORCE_OPENING[c], which the classifier chooses and
    the decoder does not generate. Tokens up to the end of that phrase are the opening; the rest (side, zone, severity,
    qualifiers) are the continuation, which is what the explanation contributes."""
    enc = tok([t + (tok.eos_token or "") for t in texts], padding="max_length", truncation=True, max_length=max_len,
              return_offsets_mapping=True, return_tensors="pt")
    op = torch.zeros_like(tm)
    for i, t in enumerate(texts):
        sp = positive_sentence_spans(t, classes)
        for c, name in enumerate(classes):
            phrase = FORCE_OPENING.get(name, "")
            for (s0, e0) in sp[name]:
                seg = t[s0:e0]
                cut = s0 + seg.index(phrase) + len(phrase) if phrase and phrase in seg else s0
                for j, (a, b) in enumerate(enc["offset_mapping"][i].tolist()):
                    if b > a and a >= s0 and b <= min(e0, cut):
                        op[i, c, j] = 1
    op = op * tm
    return op, tm - op


def remove_topk(x, dist, frac, g, blurred):
    k = max(1, int(round(frac * dist.shape[-1])))
    keep = torch.ones_like(dist)
    keep.scatter_(1, dist.topk(k, -1).indices, 0.0)
    return perturb(x, keep, g, blurred)


def remove_random(x, n_patches, frac, g, blurred, gen, keys=None, rep=0):
    """Random removal of the same area. With keys=(image index, class index) the mask is drawn from a generator seeded by
    the pair and the replicate, so the same pair receives the same control in every run and in every model; without keys the
    draws come from one generator consumed in pair order, which is what the reported runs used."""
    k = max(1, int(round(frac * n_patches)))
    keep = torch.ones(x.shape[0], n_patches, device=x.device)
    for i in range(x.shape[0]):
        if keys is None:
            perm = torch.randperm(n_patches, generator=gen)
        else:
            gi = torch.Generator().manual_seed(int((keys[0][i] * 131071 + keys[1][i] * 8191 + rep * 7919) % (2 ** 31 - 1)))
            perm = torch.randperm(n_patches, generator=gi)
        keep[i, perm[:k].to(x.device)] = 0
    return perturb(x, keep, g, blurred)


FORCE_OPENING = {
    "atelectasis": "There is atelectasis", "cardiomegaly": "The heart is enlarged", "consolidation": "There is consolidation",
    "edema": "There is pulmonary edema", "pleural_effusion": "There is a pleural effusion", "pneumonia": "There is pneumonia",
    "pneumothorax": "There is a pneumothorax", "nodule_mass": "There is a pulmonary nodule", "lung_opacity": "There is an airspace opacity",
    "emphysema": "There is emphysema", "fibrosis": "There is pulmonary fibrosis", "pleural_thickening": "There is pleural thickening",
    "hernia": "There is a hiatal hernia"}


@torch.no_grad()
def forced_explanations(model, tok, mem, prob, thresholds, classes, k, device, max_new=28):
    """Finding-conditioned explanation: for each finding the classifier flags (at most k per image, those furthest above their
    threshold), the decoder completes a sentence that opens with the finding's canonical phrase. The image-dependent content
    (side, zone, severity) and the visual grounding come from the decoder; the classifier decides which findings are explained."""
    B = prob.shape[0]
    margin = prob / np.maximum(thresholds[None], 1e-6)
    chosen = {i: sorted([c for c in range(len(classes)) if prob[i, c] >= thresholds[c]], key=lambda c: -margin[i, c])[:k] for i in range(B)}
    sent = {}
    for c, name in enumerate(classes):
        ii = [i for i in range(B) if c in chosen[i]]
        if not ii:
            continue
        fid = tok(FORCE_OPENING[name], add_special_tokens=False)["input_ids"]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids, am = model.decoder.generate_ids(mem[ii], max_new, force_ids=fid)
        txt = tok.batch_decode(ids * am + (1 - am) * tok.eos_token_id, skip_special_tokens=True)
        for i, t in zip(ii, txt):
            t = t.strip()
            sent[(i, c)] = (t.split(".")[0].strip() + ".") if "." in t else t + "."
    out = []
    for i in range(B):
        ss = [sent[(i, c)] for c in sorted(chosen[i])]
        out.append(" ".join(ss) if ss else "No acute cardiopulmonary abnormality.")
    return out


def make_infill(x, op):
    """Removal baseline: blur (the regulariser's operator), flat image mean, or Gaussian noise with the image's mean and std."""
    if op == "mean":
        return x.mean((1, 2, 3), keepdim=True).expand_as(x).contiguous()
    if op == "noise":
        gn = torch.Generator(device=x.device).manual_seed(0)
        return x.mean((1, 2, 3), keepdim=True) + x.std((1, 2, 3), keepdim=True) * torch.randn(x.shape, device=x.device, generator=gn)
    return gaussian_blur(x, sigma=x.shape[-1] / 40)


@torch.no_grad()
def run_set(model, cfg, s, device, thresholds, out_dir, args):
    classes = cfg["data"]["classes"]
    ds = ManifestDataset(s["manifest"], classes, s.get("split"), cfg["data"]["image_size"], s.get("root", ""),
                         train=False, target_mode=cfg["data"].get("target_mode", "evidence"), max_items=s.get("max_items"))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=Collator(None, classes))
    tok = model.tokenizer
    has_dec = model.decoder is not None and not args.no_text
    mem_g = model.mem_grid
    frac = args.del_frac
    gen = torch.Generator().manual_seed(0)
    ops = [o.strip() for o in args.infill_multi.split(",")] if args.infill_multi else [args.infill]
    Y, MK, P, G, rows, loc, di = [], [], [], [], [], [], []
    faith = {o: [] for o in ops}
    keep_maps = {"idx": [], "evidence": [], "gradcam": [], "text": [], "cls": []}
    abn_col = "y_abnormal" if "y_abnormal" in ds.df.columns else None
    for bi, b in enumerate(dl):
        x = b["image"].to(device)
        B = x.shape[0]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.encode(x)
            logits, phi, beta = model.classify(z)
        logits, phi = logits.float(), phi.float()
        prob = torch.sigmoid(logits)
        g = int(math.isqrt(phi.shape[-1]))
        a_full = evidence_distribution(phi)
        a_mem = pool_distribution(a_full, g, mem_g)
        pred = (prob.cpu().numpy() >= thresholds[None]).astype(int)
        texts = [""] * B
        free_texts = None
        if has_dec:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mem = model.memory(z)
                ids, am = model.decoder.generate_ids(mem, args.max_new_tokens)
            texts = [t.strip() for t in tok.batch_decode(ids * am + (1 - am) * tok.eos_token_id, skip_special_tokens=True)]
            if args.decode == "forced":
                free_texts = texts
                texts = forced_explanations(model, tok, mem, prob.cpu().numpy(), args.th_force if args.th_force is not None else thresholds, classes, args.force_k, device)
        elif args.template_text and args.th_force is not None:
            # gated template: the findings, the cap and the opening phrase of finding-conditioned decoding, without the decoder's completion
            pb = prob.cpu().numpy()
            mg = pb / np.maximum(args.th_force[None], 1e-6)
            texts = []
            for i in range(B):
                ch = sorted([c for c in range(len(classes)) if pb[i, c] >= args.th_force[c]], key=lambda c: -mg[i, c])[:args.force_k]
                texts.append(" ".join(FORCE_OPENING[classes[c]] + "." for c in sorted(ch)) or "No acute cardiopulmonary abnormality.")
        elif args.template_text:
            texts = [" ".join(f"There is {DISPLAY[c]}." for c, p in zip(classes, pr) if p) or
                     "No acute cardiopulmonary abnormality." for pr in pred]
        ment = np.array([mentioned_findings(t, classes) for t in texts])
        b_txt = valid = None
        if has_dec:
            tid, tam, tm = token_masks(tok, texts, classes, args.max_new_tokens + 8)
            tm_op, tm_co = (opening_masks(tok, texts, classes, tm, args.max_new_tokens + 8)
                            if args.ecs_split and args.decode == "forced" else (None, None))
            tid, tam, tm = tid.to(device), tam.to(device), tm.to(device)
            if tm_op is not None:
                tm_op, tm_co = tm_op.to(device), tm_co.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                grd = model.decoder.ground(mem, tid, tam)
            b_txt, valid = text_grounding(grd.float(), tm)

        # ---------------- faithfulness: text-guided deletion (TGD) and explanation counterfactual sensitivity (ECS)
        # The explanation, its grounding and the eligible pairs are fixed once per batch; every removal operator in `ops` is then
        # applied to exactly those objects. ECS compares evidence removal with the mean over args.n_random_ecs random removals.
        if has_dec:
            pairs = [(i, c) for i in range(B) for c in range(len(classes)) if valid[i, c] and pred[i, c]][: args.max_pairs_per_batch]
            if pairs:
                ii = torch.tensor([p[0] for p in pairs], device=device)
                cc = torch.tensor([p[1] for p in pairs], device=device)
                xs = x[ii]

                def score(xx, chunk=64):
                    out = []
                    for a0 in range(0, len(xx), chunk):
                        sl = slice(a0, a0 + chunk)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            lg = model.classify(model.encode(xx[sl]))[0].float()
                        out.append(lg[torch.arange(lg.shape[0]), cc[sl]])
                    return torch.cat(out)
                s0 = logits[ii, cc]
                keys = ((np.array([int(b["idx"][p[0]]) for p in pairs]), np.array([p[1] for p in pairs]))
                        if args.rand_key else None)
                # ECS: log-likelihood of the sentence(s) asserting c, before vs after removing c's evidence
                tid_p, tam_p, tm_p = tid[ii], tam[ii], tm[ii, cc]
                mk_op = tm_op[ii, cc] if tm_op is not None else None
                mk_co = tm_co[ii, cc] if tm_co is not None else None

                def sent_lp(xx, chunk=8):
                    # the vocabulary projection dominates memory, so the pairs are scored in chunks
                    parts = []
                    for a0 in range(0, len(xx), chunk):
                        sl = slice(a0, a0 + chunk)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            m2 = model.memory(model.encode(xx[sl]))
                            lp = model.decoder.token_logprob(m2, tid_p[sl], tam_p[sl]).float()
                        row = [(lp * tm_p[sl]).sum(1) / tm_p[sl].sum(1).clamp_min(1)]
                        for mk in (mk_op, mk_co):
                            mk = mk[sl] if mk is not None else None
                            row.append((lp * mk).sum(1) / mk.sum(1).clamp_min(1) if mk is not None else None)
                        parts.append(row)
                    return [torch.cat([p[k] for p in parts]) if parts[0][k] is not None else None for k in range(3)]
                lp0, lp0_op, lp0_co = sent_lp(xs)
                js = js_div(a_mem[ii, cc], b_txt[ii, cc])
                for op in ops:
                    bl = make_infill(x, op)[ii]
                    d_text = s0 - score(remove_topk(xs, b_txt[ii, cc], frac, mem_g, bl))
                    d_evid = s0 - score(remove_topk(xs, a_mem[ii, cc], frac, mem_g, bl))
                    d_rand = torch.stack([s0 - score(remove_random(xs, mem_g * mem_g, frac, mem_g, bl, gen, keys, r))
                                          for r in range(args.n_random)]).mean(0)
                    lp_ev, lpe_op, lpe_co = sent_lp(remove_topk(xs, a_mem[ii, cc], frac, mem_g, bl))
                    rnd = [sent_lp(remove_random(xs, mem_g * mem_g, frac, mem_g, bl, gen, keys, args.n_random + r))
                           for r in range(args.n_ecs_masks)]
                    mean_lp = [torch.stack([m[k] for m in rnd]).mean(0) if rnd[0][k] is not None else None for k in range(3)]
                    lp_rd, lpr_op, lpr_co = mean_lp
                    ecs_masks = torch.stack([lp0 - m[0] for m in rnd])           # one ECS reference per mask
                    for k, (i, c) in enumerate(pairs):
                        row = {"img": int(b["idx"][i]), "cls": classes[c], "d_text": float(d_text[k]), "d_evid": float(d_evid[k]),
                               "d_rand": float(d_rand[k]), "ecs_evid": float(lp0[k] - lp_ev[k]),
                               "ecs_rand": float(lp0[k] - lp_rd[k]), "js": float(js[k])}
                        if args.n_ecs_masks > 1:
                            row["ecs_rand_first"] = float(ecs_masks[0, k])            # the single-mask reference
                            row["ecs_rand_sd"] = float(ecs_masks[:, k].std())         # spread over the random masks
                        if mk_co is not None:
                            row.update({"n_tok": int(tm_p[k].sum()), "n_open": int(mk_op[k].sum()), "n_cont": int(mk_co[k].sum()),
                                        "ecs_evid_open": float(lp0_op[k] - lpe_op[k]), "ecs_rand_open": float(lp0_op[k] - lpr_op[k]),
                                        "ecs_evid_cont": float(lp0_co[k] - lpe_co[k]), "ecs_rand_cont": float(lp0_co[k] - lpr_co[k])})
                        faith[op].append(row)

        # ---------------- localisation against boxes
        for i in range(B):
            bx = b["boxes"][i].numpy()
            if len(bx) == 0:
                continue
            for c in sorted(set(bx[:, 0].astype(int).tolist())):
                cb = bx[bx[:, 0] == c]
                if c < 0:   # 'abnormal' boxes (e.g. TB): probability-weighted aggregate over all findings
                    wts = prob[i][:, None]
                    maps = {"evidence": (wts * F.relu(phi[i])).sum(0).cpu().numpy(),
                            "attention": (wts * beta[i].float()).sum(0).cpu().numpy()}
                    with torch.enable_grad():
                        gc = sum(float(prob[i, k]) * gradcam_tokens(model, x[i:i + 1], torch.tensor([k], device=device))[0].float()
                                 for k in range(len(classes)))
                    maps["gradcam"] = gc.cpu().numpy()
                    if b_txt is not None and valid[i].any():
                        maps["text"] = b_txt[i][valid[i]].mean(0).cpu().numpy()
                    cname = "abnormal"
                else:
                    cname = classes[c]
                    maps = {"evidence": F.relu(phi[i, c]).cpu().numpy(), "attention": beta[i, c].float().cpu().numpy()}
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        maps["gradcam"] = gradcam_tokens(model, x[i:i + 1], torch.tensor([c], device=device))[0].float().cpu().numpy()
                    if args.ig and len(loc) < args.ig_max:
                        maps["ig"] = integrated_gradients(model, x[i:i + 1], torch.tensor([c], device=device), steps=args.ig_steps)[0].float().cpu().numpy()
                    if b_txt is not None and valid[i, c]:
                        maps["text"] = b_txt[i, c].cpu().numpy()
                for mname, mp in maps.items():
                    r = localisation_scores(mp, cb, cfg["data"]["image_size"])
                    if r:
                        loc.append({"img": int(b["idx"][i]), "cls": cname, "map": mname,
                                    "predicted": int(pred[i, c]) if c >= 0 else int(pred[i].any()), **r})
                if len(keep_maps["idx"]) < args.save_maps:
                    keep_maps["idx"].append(int(b["idx"][i]))
                    keep_maps["cls"].append(c)
                    keep_maps["evidence"].append(maps["evidence"])
                    keep_maps["gradcam"].append(maps["gradcam"])
                    keep_maps["text"].append(maps.get("text", np.full(mem_g * mem_g, np.nan)))

        # ---------------- deletion / insertion for the top predicted class (subsample)
        if args.del_ins and bi % args.del_ins_every == 0:
            c = prob.argmax(1)
            ar = torch.arange(B, device=device)

            def pfn(xx):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    return torch.sigmoid(model.classify(model.encode(xx))[0].float()[ar, c])
            for mname, sal in [("evidence", F.relu(phi[ar, c])), ("gradcam", gradcam_tokens(model, x, c).float())]:
                d, ins = deletion_insertion_auc(pfn, x, sal, steps=args.del_steps)
                di += [{"map": mname, "deletion_auc": float(dd), "insertion_auc": float(ii_)} for dd, ii_ in zip(d.cpu(), ins.cpu())]

        Y.append(b["y"].numpy()), MK.append(b["y_mask"].numpy()), P.append(prob.cpu().numpy())
        G += b["patient_id"]
        for i in range(B):
            r = {"idx": int(b["idx"][i]), "patient_id": b["patient_id"][i], "generated": texts[i], "reference": b["texts"][i]}
            if free_texts is not None:
                r["generated_free"] = free_texts[i]
            r.update({f"p_{c}": float(prob[i, k]) for k, c in enumerate(classes)})
            r.update({f"m_{c}": int(ment[i, k]) for k, c in enumerate(classes)})
            rows.append(r)
        if args.max_batches and bi + 1 >= args.max_batches:
            break

    y, mk, p = np.concatenate(Y), np.concatenate(MK), np.concatenate(P)
    per = pd.DataFrame(rows)
    summary = {"set": s["name"], "n_images": len(per)}
    summary["classification"] = classification_report(y, mk, p, classes, G, n_boot=args.n_boot, thresholds=thresholds)
    summary["ece"] = ece(y, mk, p)
    if abn_col:
        lung = [k for k, c in enumerate(classes) if FINDINGS[c]["region"] != "any"]
        score = 1 - np.prod(1 - p[:, lung], axis=1)
        ya = ds.df[abn_col].values[: len(score)].astype(float)
        from sklearn.metrics import roc_auc_score
        summary["abnormality_auroc"] = float(roc_auc_score(ya, score)) if 0 < ya.sum() < len(ya) else None
        if summary["abnormality_auroc"] is not None and args.n_boot:
            from evialign.metrics import patient_bootstrap
            bs = []
            for idx in patient_bootstrap(list(G[: len(score)]), args.n_boot):
                if 0 < ya[idx].sum() < len(idx):
                    bs.append(roc_auc_score(ya[idx], score[idx]))
            summary["abnormality_auroc_ci"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
    if has_dec or args.template_text:
        mentioned = per[[f"m_{c}" for c in classes]].values
        predb = (p >= thresholds[None]).astype(int)
        summary["consistency"] = prediction_explanation_consistency(predb, mentioned, y, mk)
        refs = per["reference"].fillna("").tolist()
        if any(len(r) for r in refs):
            pairs = [(r, h) for r, h in zip(refs, per["generated"]) if r]
            summary["text"] = {"bleu4": corpus_bleu4(*zip(*pairs)), "rougeL": rouge_l(*zip(*pairs)), "n": len(pairs)}
    for oi, op in enumerate(ops):
        if not faith[op]:
            continue
        f = pd.DataFrame(faith[op])
        f.to_csv(os.path.join(out_dir, f"faithfulness_{s['name']}.csv" if oi == 0 else f"faithfulness_{s['name']}_{op}.csv"), index=False)
        if len(ops) > 1:
            f.to_csv(os.path.join(out_dir, f"faithfulness_{s['name']}_{op}.csv"), index=False)
        denom = (f["d_evid"] - f["d_rand"]).mean()
        fs = {
            "n_pairs": len(f), "tgd_drop_text": f["d_text"].mean(), "tgd_drop_evidence": f["d_evid"].mean(),
            "tgd_drop_random": f["d_rand"].mean(),
            "tgd_normalised": float((f["d_text"].mean() - f["d_rand"].mean()) / denom) if abs(denom) > 1e-8 else None,
            "tgd_win_rate_vs_random": float((f["d_text"] > f["d_rand"]).mean()),
            "ecs_evidence": f["ecs_evid"].mean(), "ecs_random": f["ecs_rand"].mean(),
            "ecs_gap": float((f["ecs_evid"] - f["ecs_rand"]).mean()), "js_text_vs_evidence": f["js"].mean()}
        rng = np.random.default_rng(0)
        ex = (f["d_text"] - f["d_rand"]).values
        eg = (f["ecs_evid"] - f["ecs_rand"]).values
        bs = [rng.integers(0, len(f), len(f)) for _ in range(1000)]
        fs["tgd_excess_drop"] = float(ex.mean())
        fs["tgd_excess_drop_ci"] = [float(v) for v in np.percentile([ex[b].mean() for b in bs], [2.5, 97.5])]
        fs["ecs_gap_ci"] = [float(v) for v in np.percentile([eg[b].mean() for b in bs], [2.5, 97.5])]
        if "ecs_rand_first" in f:
            fs["ecs_gap_single_mask"] = float((f["ecs_evid"] - f["ecs_rand_first"]).mean())
        if oi == 0:
            summary["faithfulness"] = fs
        if len(ops) > 1:
            summary.setdefault("faithfulness_by_operator", {})[op] = fs
    if loc:
        l = pd.DataFrame(loc)
        l.to_csv(os.path.join(out_dir, f"localisation_{s['name']}.csv"), index=False)
        summary["localisation"] = {m: {"hit": d["hit"].mean(), "energy": d["energy"].mean(),
                                       "mass_area_ratio": d["mass_area_ratio"].median(), "n": len(d)}
                                   for m, d in l.groupby("map")}
        summary["localisation_per_class"] = {f"{m}|{c}": {"hit": d["hit"].mean(), "energy": d["energy"].mean(), "n": len(d)}
                                             for (m, c), d in l.groupby(["map", "cls"])}
    if di:
        d = pd.DataFrame(di)
        summary["deletion_insertion"] = {m: {"deletion_auc": v["deletion_auc"].mean(), "insertion_auc": v["insertion_auc"].mean()}
                                         for m, v in d.groupby("map")}
    per.to_csv(os.path.join(out_dir, f"per_image_{s['name']}.csv"), index=False)
    if keep_maps["idx"]:
        np.savez_compressed(os.path.join(out_dir, f"maps_{s['name']}.npz"), **{k: np.array(v) for k, v in keep_maps.items()})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--del_frac", type=float, default=0.15)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--decode", default="free", choices=["free", "forced"],
                    help="free: greedy free text; forced: one sentence per classifier-flagged finding, opened by a canonical phrase")
    ap.add_argument("--force_spec", type=float, default=0.0,
                    help="forced decoding only: explain findings above the validation threshold that gives this specificity (e.g. 0.9); 0 uses the F1 thresholds")
    ap.add_argument("--force_k", type=int, default=3, help="at most this many flagged findings are explained per image (forced decoding)")
    ap.add_argument("--infill", default="blur", choices=["blur", "mean", "noise"],
                    help="removal baseline for TGD/ECS; the training regulariser uses blur, so mean/noise are a check that is not trained for")
    ap.add_argument("--infill_multi", default="",
                    help="comma-separated removal operators (blur, mean, noise) applied to one and the same generated explanation, "
                         "grounding and set of eligible pairs; the first is the primary one and keeps the unsuffixed output files")
    ap.add_argument("--n_ecs_masks", type=int, default=1,
                    help="number of independent random removals averaged to form the ECS reference (1 reproduces the single-mask reference of the first runs)")
    ap.add_argument("--max_pairs_per_batch", type=int, default=64,
                    help="cap on scored (image, finding) pairs per batch, taken in image-then-finding order; 512 scores every eligible pair")
    ap.add_argument("--rand_key", action="store_true",
                    help="draw each random-removal mask from a generator seeded by (image, finding, replicate), so every model and every run gets the same control for a shared pair")
    ap.add_argument("--ecs_split", action="store_true",
                    help="forced decoding only: also record ECS over the forced opening phrase and over the decoder's continuation separately")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--ig", action="store_true")
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--ig_max", type=int, default=500)
    ap.add_argument("--del_ins", action="store_true")
    ap.add_argument("--del_ins_every", type=int, default=10)
    ap.add_argument("--del_steps", type=int, default=10)
    ap.add_argument("--save_maps", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=0, help="evaluate only the first N batches of each set (fixed, deterministic subset)")
    ap.add_argument("--resume", action="store_true", help="skip sets already present in <out>/summary.json")
    ap.add_argument("--no_text", action="store_true", help="skip the decoder (classifier-only evaluation)")
    ap.add_argument("--template_text", action="store_true", help="template explanations from predictions (baseline)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = EviAlign(cfg, len(cfg["data"]["classes"]))
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    os.makedirs(args.out, exist_ok=True)
    with open(args.sets) as f:
        sets = yaml.safe_load(f)["sets"]

    # thresholds on the internal validation split (F1-optimal per class)
    th_path = os.path.join(args.out, "thresholds.json")
    ths_path = os.path.join(args.out, "thresholds_spec.json")
    args.th_force = None
    need_val = (not os.path.exists(th_path)) or (args.force_spec > 0 and not os.path.exists(ths_path))
    if need_val:
        D = cfg["data"]
        va = ManifestDataset(D.get("val_manifest", D["train_manifest"]), D["classes"], D.get("val_split", "val"),
                             D["image_size"], D.get("root", ""), max_items=D.get("max_val_items"))
        dl = DataLoader(va, batch_size=args.batch_size, num_workers=args.workers, collate_fn=Collator(None, D["classes"]))
        Y, MK, P = [], [], []
        with torch.no_grad():
            for i, b in enumerate(dl):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    lg = model.classify(model.encode(b["image"].to(device)))[0]
                P.append(torch.sigmoid(lg.float()).cpu().numpy()), Y.append(b["y"].numpy()), MK.append(b["y_mask"].numpy())
                if args.max_batches and i + 1 >= args.max_batches:
                    break
        Y, MK, P = np.concatenate(Y), np.concatenate(MK), np.concatenate(P)
        if not os.path.exists(th_path):
            json.dump(f1_thresholds(Y, MK, P).tolist(), open(th_path, "w"))
        if args.force_spec > 0:       # high-specificity operating point used only to choose which findings get an explanation
            ths = np.full(Y.shape[1], 0.5)
            for c in range(Y.shape[1]):
                neg = (MK[:, c] > 0) & (Y[:, c] < 0.5)
                if neg.sum() > 20:
                    ths[c] = float(np.quantile(P[neg, c], args.force_spec))
            json.dump(ths.tolist(), open(ths_path, "w"))
    th = np.array(json.load(open(th_path)))
    if args.force_spec > 0:
        args.th_force = np.array(json.load(open(ths_path)))
    allsum = {}
    sp = os.path.join(args.out, "summary.json")
    if args.resume and os.path.exists(sp):
        allsum = json.load(open(sp))
    for s in sets:
        if not os.path.exists(s["manifest"]):
            print(f"== {s['name']} (manifest {s['manifest']} not found, skipped)", flush=True)
            continue
        if s["name"] in allsum and args.resume:
            print(f"== {s['name']} (done, skipped)", flush=True)
            continue
        print(f"== {s['name']}", flush=True)
        allsum[s["name"]] = run_set(model, cfg, s, device, th, args.out, args)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(allsum, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(json.dumps({k: v.get("classification", {}).get("macro") for k, v in allsum.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
