"""Train EviAlign-CXR.

Single GPU:   python train.py --config configs/main.yaml
Two GPUs:     torchrun --nproc_per_node 2 train.py --config configs/main.yaml
Override:     python train.py --config configs/main.yaml loss.w_align=0 output_dir=runs/no_align
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from evialign.data import Collator, ManifestDataset
from evialign.metrics import classification_report
from evialign.models.model import EviAlign
from evialign.objective import Objective


def load_cfg(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_cfg(os.path.join(os.path.dirname(path), cfg.pop("base")), [])
        cfg = deep_update(base, cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return cfg


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = deep_update(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def weights_for_epoch(cfg: dict, epoch: float) -> dict:
    L, S = cfg["loss"], cfg["train"].get("stages", {})
    cls_only = S.get("cls_only_epochs", 0)
    a0, ramp = S.get("align_start_epoch", cls_only), max(S.get("align_ramp_epochs", 1), 1e-6)
    has_dec = cfg["model"].get("decoder", {}).get("type", "none") != "none"
    w = {"cls": L.get("w_cls", 1.0), "lm": 0.0, "align": 0.0, "inst": 0.0,
         "anat": L.get("w_anat", 0.0) if epoch >= S.get("anat_start_epoch", 0) else 0.0, "suff": 0.0, "comp": 0.0}
    if has_dec and epoch >= cls_only:
        w["lm"] = L.get("w_lm", 1.0)
    r = min(max((epoch - a0) / ramp, 0.0), 1.0) if epoch >= a0 else 0.0
    if has_dec:
        w["align"], w["inst"] = r * L.get("w_align", 0.0), r * L.get("w_inst", 0.0)
    if epoch >= S.get("sc_start_epoch", cls_only):
        w["suff"], w["comp"] = L.get("w_suff", 0.0), L.get("w_comp", 0.0)
    return w


def param_groups(model, cfg):
    T = cfg["train"]
    enc, head, dec = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith("encoder.") else head if n.startswith("head.") else dec).append(p)
    return [{"params": enc, "lr": T["lr_encoder"]}, {"params": head, "lr": T["lr_head"]},
            {"params": dec, "lr": T.get("lr_decoder", 1e-4)}]


@torch.no_grad()
def validate(model, loader, device, classes, amp_dtype, max_batches=None):
    model.eval()
    Y, Mk, P, G = [], [], [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model.classify(model.encode(b["image"].to(device)))[0]
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        Y.append(b["y"].numpy())
        Mk.append(b["y_mask"].numpy())
        G += b["patient_id"]
    model.train()
    return classification_report(np.concatenate(Y), np.concatenate(Mk), np.concatenate(P), classes, G, n_boot=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.overrides)

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("seed", 0) + rank
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    out_dir = cfg["output_dir"]
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    D, T = cfg["data"], cfg["train"]
    classes = D["classes"]
    model = EviAlign(cfg, len(classes)).to(device)
    tok = model.tokenizer
    collate = Collator(tok, classes, D.get("max_text_len", 128))
    tr = ManifestDataset(D["train_manifest"], classes, D.get("train_split", "train"), D["image_size"], D.get("root", ""),
                         train=True, target_mode=D.get("target_mode", "evidence"), max_items=D.get("max_train_items"))
    va = ManifestDataset(D.get("val_manifest", D["train_manifest"]), classes, D.get("val_split", "val"), D["image_size"],
                         D.get("root", ""), train=False, max_items=D.get("max_val_items"))
    sampler = DistributedSampler(tr, shuffle=True) if ddp else None
    tl = DataLoader(tr, batch_size=T["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=D.get("num_workers", 8),
                    collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=D.get("num_workers", 8) > 0)
    vl = DataLoader(va, batch_size=T.get("eval_batch_size", T["batch_size"]), shuffle=False,
                    num_workers=D.get("num_workers", 8), collate_fn=Collator(None, classes))

    # positive-class weighting from the training labels (capped)
    ys = tr.df[[f"y_{c}" for c in classes]].replace(-1, 1).fillna(0).clip(lower=0).values.astype(float)
    pos = ys.mean(0).clip(1e-4, None)
    pw = torch.tensor(np.minimum((1 - pos) / pos, cfg["loss"].get("pos_weight_cap", 10.0)), dtype=torch.float32)
    obj = Objective(model, classes, cfg["loss"], pw).to(device)
    if ddp:
        obj = torch.nn.parallel.DistributedDataParallel(obj, device_ids=[local], find_unused_parameters=True)
    core = model

    opt = torch.optim.AdamW(param_groups(core, cfg), weight_decay=T.get("weight_decay", 0.05))
    steps_per_epoch = len(tl) // T.get("grad_accum", 1)
    total = steps_per_epoch * T["epochs"]
    warm = T.get("warmup_steps", 500)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0))))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(T.get("amp", "bf16")) if device.type == "cuda" else None

    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)

    log = open(os.path.join(out_dir, "log.jsonl"), "a") if rank == 0 else None
    step = 0
    for epoch in range(start_epoch, T["epochs"]):
        if sampler:
            sampler.set_epoch(epoch)
        t0 = time.time()
        for it, b in enumerate(tl):
            frac_epoch = epoch + it / max(len(tl), 1)
            w = weights_for_epoch(cfg, frac_epoch)
            if cfg["loss"].get("sc_every", 1) > 1 and (step % cfg["loss"]["sc_every"]) != 0:
                w["suff"] = w["comp"] = 0.0
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                L = obj(b, w)
            (L["total"] / T.get("grad_accum", 1)).backward()
            if (it + 1) % T.get("grad_accum", 1) == 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), T.get("clip", 1.0))
                opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
            if rank == 0 and it % T.get("log_every", 50) == 0:
                rec = {"epoch": round(frac_epoch, 3), "step": step, **{k: float(v) for k, v in L.items() if not k.startswith("_")},
                       **{f"w_{k}": v for k, v in w.items()}, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
                       "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device.type == "cuda" else 0.0}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"), log.flush()
            if T.get("max_steps_per_epoch") and it + 1 >= T["max_steps_per_epoch"]:
                break
        if rank == 0:
            rep = validate(core, vl, device, classes, amp_dtype, T.get("val_max_batches"))
            auc = rep["macro"]["auroc"]
            rec = {"epoch": epoch, "val_macro_auroc": auc, "val_per_class": {k: v["auroc"] for k, v in rep["per_class"].items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"), log.flush()
            ck = {"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                  "best": best, "cfg": cfg}
            torch.save(ck, os.path.join(out_dir, "last.pt"))
            # model selection on the classification task only, so that faithfulness metrics stay untouched
            S = T.get("stages", {})
            sel_from = T.get("select_from_epoch", S.get("align_start_epoch", 0) + S.get("align_ramp_epochs", 0))
            if auc == auc and auc > best and epoch >= sel_from:
                best = auc
                torch.save(ck, os.path.join(out_dir, "best.pt"))
        if ddp:
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
