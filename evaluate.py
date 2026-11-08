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


def remove_topk(x, dist, frac, g, blurred):
    k = max(1, int(round(frac * dist.shape[-1])))
    keep = torch.ones_like(dist)
    keep.scatter_(1, dist.topk(k, -1).indices, 0.0)
    return perturb(x, keep, g, blurred)


def remove_random(x, n_patches, frac, g, blurred, gen):
    k = max(1, int(round(frac * n_patches)))
    keep = torch.ones(x.shape[0], n_patches, device=x.device)
    for i in range(x.shape[0]):
        keep[i, torch.randperm(n_patches, generator=gen)[:k].to(x.device)] = 0
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
    Y, MK, P, G, rows, loc, faith, di = [], [], [], [], [], [], [], []
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
        elif args.template_text:
            texts = [" ".join(f"There is {DISPLAY[c]}." for c, p in zip(classes, pr) if p) or
                     "No acute cardiopulmonary abnormality." for pr in pred]
        ment = np.array([mentioned_findings(t, classes) for t in texts])
        b_txt = valid = None
        if has_dec:
            tid, tam, tm = token_masks(tok, texts, classes, args.max_new_tokens + 8)
            tid, tam, tm = tid.to(device), tam.to(device), tm.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                grd = model.decoder.ground(mem, tid, tam)
            b_txt, valid = text_grounding(grd.float(), tm)
        blurred = gaussian_blur(x, sigma=x.shape[-1] / 40)
        if args.infill == "mean":      # removal baseline unseen in training: flat image mean
            blurred = x.mean((1, 2, 3), keepdim=True).expand_as(x).contiguous()
        elif args.infill == "noise":   # removal baseline unseen in training: Gaussian noise with the image's mean and std
            gn = torch.Generator(device=x.device).manual_seed(0)
            blurred = x.mean((1, 2, 3), keepdim=True) + x.std((1, 2, 3), keepdim=True) * torch.randn(x.shape, device=x.device, generator=gn)

        # ---------------- faithfulness: text-guided deletion (TGD) and explanation counterfactual sensitivity (ECS)
        if has_dec:
            pairs = [(i, c) for i in range(B) for c in range(len(classes)) if valid[i, c] and pred[i, c]][: args.max_pairs_per_batch]
            if pairs:
                ii = torch.tensor([p[0] for p in pairs], device=device)
                cc = torch.tensor([p[1] for p in pairs], device=device)
                xs, bl = x[ii], blurred[ii]

                def score(xx):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        return model.classify(model.encode(xx))[0].float()[torch.arange(len(pairs)), cc]
                s0 = logits[ii, cc]
                d_text = s0 - score(remove_topk(xs, b_txt[ii, cc], frac, mem_g, bl))
                d_evid = s0 - score(remove_topk(xs, a_mem[ii, cc], frac, mem_g, bl))
                d_rand = torch.stack([s0 - score(remove_random(xs, mem_g * mem_g, frac, mem_g, bl, gen)) for _ in range(args.n_random)]).mean(0)
                # ECS: log-likelihood of the sentence(s) asserting c, before vs after removing c's evidence
                tid_p, tam_p, tm_p = tid[ii], tam[ii], tm[ii, cc]
                def sent_lp(xx):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        m2 = model.memory(model.encode(xx))
                        lp = model.decoder.token_logprob(m2, tid_p, tam_p).float()
                    return (lp * tm_p).sum(1) / tm_p.sum(1).clamp_min(1)
                lp0 = sent_lp(xs)
                lp_ev = sent_lp(remove_topk(xs, a_mem[ii, cc], frac, mem_g, bl))
                lp_rd = sent_lp(remove_random(xs, mem_g * mem_g, frac, mem_g, bl, gen))
                js = js_div(a_mem[ii, cc], b_txt[ii, cc])
                for k, (i, c) in enumerate(pairs):
                    faith.append({"img": int(b["idx"][i]), "cls": classes[c], "d_text": float(d_text[k]), "d_evid": float(d_evid[k]),
                                  "d_rand": float(d_rand[k]), "ecs_evid": float(lp0[k] - lp_ev[k]),
                                  "ecs_rand": float(lp0[k] - lp_rd[k]), "js": float(js[k])})

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
    if has_dec or args.template_text:
        mentioned = per[[f"m_{c}" for c in classes]].values
        predb = (p >= thresholds[None]).astype(int)
        summary["consistency"] = prediction_explanation_consistency(predb, mentioned, y, mk)
        refs = per["reference"].fillna("").tolist()
        if any(len(r) for r in refs):
            pairs = [(r, h) for r, h in zip(refs, per["generated"]) if r]
            summary["text"] = {"bleu4": corpus_bleu4(*zip(*pairs)), "rougeL": rouge_l(*zip(*pairs)), "n": len(pairs)}
    if faith:
        f = pd.DataFrame(faith)
        f.to_csv(os.path.join(out_dir, f"faithfulness_{s['name']}.csv"), index=False)
        denom = (f["d_evid"] - f["d_rand"]).mean()
        summary["faithfulness"] = {
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
        summary["faithfulness"]["tgd_excess_drop"] = float(ex.mean())
        summary["faithfulness"]["tgd_excess_drop_ci"] = [float(v) for v in np.percentile([ex[b].mean() for b in bs], [2.5, 97.5])]
        summary["faithfulness"]["ecs_gap_ci"] = [float(v) for v in np.percentile([eg[b].mean() for b in bs], [2.5, 97.5])]
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
    ap.add_argument("--max_pairs_per_batch", type=int, default=64)
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
