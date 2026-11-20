"""Figure 2: qualitative cases for the main model (SVG; black text, compact layout).

Cases are drawn at random (fixed seed) within categories, from the evaluation outputs of the main run:
  agreement   classifier evidence and explanation grounding both peak on the radiologist box of an asserted finding
  grounding miss   the explanation grounding peaks outside the box of an asserted finding
  false assertion  an asserted finding that is negative in the labels (NIH), grounded as closely to the evidence as the true ones
Columns: radiograph with boxes, classifier evidence, explanation grounding of the asserted finding, Grad-CAM.
The generated explanation is printed under each row. Selection needs only the CSVs written by evaluate.py; rendering needs the checkpoint and images.

python tools/make_qualitative.py --run runs/main --sets_yaml configs/eval_sets.yaml --out results/figures
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import textwrap

import numpy as np
import pandas as pd
import yaml

PLAN = [("nih_test", "agreement", 2), ("nih_test", "grounding miss", 1), ("nih_test", "false assertion", 1),
        ("chestxdet", "agreement", 1), ("chestxdet", "grounding miss", 1)]
LABEL = {"nih_test": "NIH", "chestxdet": "ChestX-Det"}


def select_cases(run, sets, seed=0):
    import json
    rng = np.random.default_rng(seed)
    S = {s["name"]: s for s in sets}
    gate = json.load(open(os.path.join(run, "eval", "thresholds_spec.json")))
    MAN = {}

    def manifest(name):
        if name not in MAN:
            m = pd.read_csv(S[name]["manifest"], low_memory=False)
            MAN[name] = m[m["split"] == S[name].get("split", "test")].reset_index(drop=True)
        return MAN[name]

    out, used = [], set()
    for set_name, cat, n in PLAN:
        ev = os.path.join(run, "eval")
        loc = pd.read_csv(os.path.join(ev, f"localisation_{set_name}.csv"))
        per = pd.read_csv(os.path.join(ev, f"per_image_{set_name}.csv")).set_index("idx")
        if cat == "false assertion":
            fa = pd.read_csv(os.path.join(ev, f"faithfulness_{set_name}.csv"))
            man = manifest(set_name)
            fa["y"] = [man.loc[int(i), "y_" + c] for i, c in zip(fa["img"], fa["cls"])]
            cand = fa[(fa["y"] == 0) & (fa["js"] <= fa["js"].median())].rename(columns={"img": "idx"})
        else:
            p = loc.pivot_table(index=["img", "cls"], columns="map", values="hit", aggfunc="first").reset_index().rename(columns={"img": "idx"})
            p = p.dropna(subset=["text", "evidence"])
            pred = loc[loc["map"] == "evidence"].set_index(["img", "cls"])["predicted"]
            p["predicted"] = [pred.get((i, c), 0) for i, c in zip(p["idx"], p["cls"])]
            p = p[p["predicted"] == 1]
            cand = p[(p["text"] == 1) & (p["evidence"] == 1)] if cat == "agreement" else p[p["text"] == 0]
        cand = cand[~cand.apply(lambda r: (set_name, int(r["idx"])) in used, axis=1)]
        cand = cand.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        seen_cls, k = set(), 0
        for _, r in cand.iterrows():
            if r["cls"] in seen_cls or int(r["idx"]) not in per.index:
                continue
            seen_cls.add(r["cls"])
            used.add((set_name, int(r["idx"])))
            y = pd.to_numeric(manifest(set_name).loc[int(r["idx"]), "y_" + r["cls"]], errors="coerce")
            classes = [c[2:] for c in per.columns if c.startswith("p_")]
            out.append({"set": set_name, "category": cat, "idx": int(r["idx"]), "cls": r["cls"],
                        "text": str(per.loc[int(r["idx"]), "generated"]),
                        "p": float(per.loc[int(r["idx"]), "p_" + r["cls"]]),
                        "gate": float(gate[classes.index(r["cls"])]),
                        "label": "not annotated" if pd.isna(y) else ("positive" if y > 0.5 else "negative")})
            k += 1
            if k == n:
                break
    return out


def render(run, cases, sets_yaml, out, max_new_tokens=96):
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from evaluate import token_masks
    from evialign.attribution import gradcam_tokens
    from evialign.data import IMAGENET_MEAN, IMAGENET_STD, ManifestDataset
    from evialign.losses import text_grounding
    from evialign.metrics import upsample_map
    from evialign.models.model import EviAlign

    plt.rcParams.update({"font.size": 7, "text.color": "black", "axes.edgecolor": "black", "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    classes = cfg["data"]["classes"]
    model = EviAlign(cfg, len(classes))
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    tok = model.tokenizer
    S = {s["name"]: s for s in yaml.safe_load(open(sets_yaml))["sets"]}
    ds = {}
    rows = []
    for cs in cases:
        n = cs["set"]
        if n not in ds:
            ds[n] = ManifestDataset(S[n]["manifest"], classes, S[n].get("split"), cfg["data"]["image_size"], S[n].get("root", ""), train=False)
        it = ds[n][cs["idx"]]
        c = classes.index(cs["cls"])
        x = it["image"][None].to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.encode(x)
            _, phi, _ = model.classify(z)
            mem = model.memory(z)
            tid, tam, tm = token_masks(tok, [cs["text"]], classes, max_new_tokens + 8)
            grd = model.decoder.ground(mem, tid.to(device), tam.to(device))
            b_txt, _ = text_grounding(grd.float(), tm.to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            gc = gradcam_tokens(model, x, torch.tensor([c], device=device))[0].float().cpu().numpy()
        maps = {"evidence": F.relu(phi[0, c].float()).cpu().numpy(), "text": b_txt[0, c].float().cpu().numpy(), "gradcam": gc}
        img = (it["image"][0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]).clamp(0, 1).numpy()
        bx = it["boxes"].numpy()
        rows.append((cs, img, maps, bx[bx[:, 0] == c]))

    k = len(rows)
    fig = plt.figure(figsize=(6.6, 1.80 * k))
    gs = fig.add_gridspec(2 * k, 4, height_ratios=[1, 0.3] * k, hspace=0.04, wspace=0.03, left=0.055, right=0.995, top=0.97, bottom=0.01)
    titles = ["Radiograph and boxes", "Classifier evidence", "Explanation grounding", "Grad-CAM"]
    for r, (cs, img, maps, bx) in enumerate(rows):
        for j, key in enumerate([None, "evidence", "text", "gradcam"]):
            ax = fig.add_subplot(gs[2 * r, j])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if key:
                m = np.clip(upsample_map(np.nan_to_num(maps[key]), img.shape[0]), 0, None)
                ax.imshow(m / max(m.max(), 1e-8), cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            else:
                for (_, x0, y0, w, h) in bx:
                    ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, ec="#00e5ff", lw=1.2))
                ax.set_ylabel(cs["category"], fontsize=6.5)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[j], fontsize=7)
        tx = fig.add_subplot(gs[2 * r + 1, :])
        tx.axis("off")
        head = (f"{LABEL[cs['set']]} · {cs['cls'].replace('_', ' ')} · p = {cs['p']:.2f} (gate {cs['gate']:.2f}) · "
                f"reference label {cs['label']}")
        tx.text(0.0, 0.78, head, va="center", ha="left", fontsize=6.3, fontweight="bold")
        tx.text(0.0, 0.28, "\n".join(textwrap.wrap(cs["text"], 150)[:2]), va="center", ha="left", fontsize=6.3)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_qualitative.svg"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets_yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select_only", action="store_true")
    a = ap.parse_args()
    sets = yaml.safe_load(open(a.sets_yaml))["sets"]
    cases = select_cases(a.run, sets, a.seed)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(cases).to_csv(os.path.join(a.out, "fig_qualitative_cases.csv"), index=False)
    for c in cases:
        print(c["set"], c["category"], c["idx"], c["cls"], "|", c["text"][:110])
    if not a.select_only:
        render(a.run, cases, a.sets_yaml, a.out)
        print("written", os.path.join(a.out, "fig_qualitative.svg"))


if __name__ == "__main__":
    main()
