"""Blinded reader package for expert adjudication of the explanations.

Each case is one (image, finding) pair that has a radiologist box, so that every case can be related to the boxes afterwards, and that
the guided model, the unguided model and the template comparator all assert. A case is one PNG with a clean radiograph and three explanation panels (A, B, C; the assignment of systems to letters is shuffled per case and kept in
key.csv). A panel shows the sentence that asserts the finding and the map the system reports for it: the explanation grounding of the
sentence for the two language-model systems and the classifier evidence for the template comparator. The reader sees no probabilities,
labels, boxes or system names. Cases are drawn at random (fixed seed), stratified by corpus and finding.

python tools/adjudication_package.py --runs runs/main runs/plain_vlm --template_run runs/classifier_only \
    --sets_yaml configs/eval_sets.yaml --n_per_set 24 --controls 18 --out adjudication
Writes <out>/cases/case_XXX.png, <out>/batches/batch_NN/ (the same images in groups for readers that accept a few images per message),
<out>/ratings_template.csv and <out>/panel_prompt.txt (for readers that are language models), and <out>/key.csv (do not give to readers).
Mismatched controls show the radiograph of one case with the sentences and overlays of another case with a different finding; their
sentence ratings should be 0 and their highlight ratings low, which tests whether a reader can detect a wrong explanation.
tools/adjudication_analysis.py summarises the completed sheets.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import textwrap

import numpy as np
import pandas as pd
import yaml

from evialign.labeler import extract_mentions

SETS = ["nih_test", "chestxdet", "rsna"]
OPACITY = {"lung_opacity", "pneumonia", "consolidation", "atelectasis"}   # findings a reader may reasonably confuse; not paired in controls
ARMS = [("guided", "runs/main", "eval", "text"), ("unguided", "runs/plain_vlm", "eval", "text"),
        ("template", "runs/classifier_only", "eval_template_gated", "evidence")]


INSTRUCTIONS = """Each image cases/case_XXX.png shows a chest radiograph and three short reports (A, B, C) for the finding named under the radiograph.
Reports are shown with a coloured overlay marking the image region that the system uses for the sentence; brighter means more weight.
Fill one row per case in ratings_template.csv (save a copy per reader as ratings_<reader>.csv and enter the reader name in 'reader').

finding_present  Judged from the radiograph alone: yes / no / uncertain.
sentence_A/B/C   0 = the sentence names the wrong finding, side or zone; 1 = the finding is right and the location is missing or partly wrong;
                 2 = the finding and its location are correct. Rate the sentence even if you judged the finding absent (then use 0).
highlight_A/B/C  0 = the bright region is not on the finding; 1 = partly on it; 2 = on it.
preferred        The report whose sentence and overlay together you would prefer to receive: A, B, C or none.
Leave the 'comment' column free. Do not look for the origin of a report; the letters are assigned at random per case."""


def sentences_by_finding(text):
    text = text if isinstance(text, str) else ""
    out = {}
    for m in extract_mentions(text):
        if m.status == 1 and m.finding not in out:
            out[m.finding] = text[m.sent_start:m.sent_end].strip()
    return out


def select(roots, n_per_set, seed):
    rng = np.random.default_rng(seed)
    cases = []
    for s in SETS:
        S = {}
        for name, run, sub, _ in ARMS:
            f = os.path.join(run, sub, f"per_image_{s}.csv")
            if not os.path.exists(f):
                break
            d = pd.read_csv(f, usecols=["idx", "generated"])
            S[name] = {int(i): sentences_by_finding(t) for i, t in zip(d.idx, d.generated)}
        if len(S) < len(ARMS):
            continue
        L = pd.read_csv(os.path.join(ARMS[0][1], ARMS[0][2], f"localisation_{s}.csv"), usecols=["img", "cls"]).drop_duplicates()
        boxed = set(zip(L.img.astype(int), L.cls))
        cand = []
        for i in set.intersection(*[set(v) for v in S.values()]):
            for c in set.intersection(*[set(v[i]) for v in S.values()]):
                if (i, c) in boxed:
                    cand.append(dict(set=s, idx=i, cls=c, **{f"text_{n}": S[n][i][c] for n in S}))
        cand = pd.DataFrame(cand)
        if cand.empty:
            continue
        # equal draw per finding where possible
        by = {c: g.sample(frac=1, random_state=int(rng.integers(1 << 31))) for c, g in cand.groupby("cls")}
        picks, k = [], 0
        while len(picks) < min(n_per_set, len(cand)):
            for c in sorted(by):
                if k < len(by[c]) and len(picks) < n_per_set:
                    picks.append(by[c].iloc[k])
            k += 1
        cases += [dict(p) for p in picks]
    return cases


def maps(run, cases, sets_yaml, which, max_new_tokens=96):
    import torch
    import torch.nn.functional as F
    from evaluate import token_masks
    from evialign.data import IMAGENET_MEAN, IMAGENET_STD, ManifestDataset
    from evialign.losses import text_grounding
    from evialign.models.model import EviAlign
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; classes = cfg["data"]["classes"]
    model = EviAlign(cfg, len(classes)); model.load_state_dict(ck["model"], strict=False); model.to(device).eval()
    S = {s["name"]: s for s in yaml.safe_load(open(sets_yaml))["sets"]}
    ds, out = {}, []
    for cs in cases:
        n = cs["set"]
        if n not in ds:
            ds[n] = ManifestDataset(S[n]["manifest"], classes, S[n].get("split"), cfg["data"]["image_size"], S[n].get("root", ""), train=False)
        it = ds[n][cs["idx"]]
        x = it["image"][None].to(device); c = classes.index(cs["cls"])
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.encode(x)
            if which == "evidence":
                m = F.relu(model.classify(z)[1][0, c].float()).cpu().numpy()
            else:
                mem = model.memory(z)
                tid, tam, tm = token_masks(model.tokenizer, [cs["sentence"]], classes, max_new_tokens + 8)
                grd = model.decoder.ground(mem, tid.to(device), tam.to(device))
                m = text_grounding(grd.float(), tm.to(device))[0][0, c].float().cpu().numpy()
        img = (it["image"][0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]).clamp(0, 1).numpy()
        out.append((img, np.nan_to_num(m)))
    return out


PANEL_PROMPT = """You are one of several independent raters in a methods study of automated chest radiograph reports. Each attached image is one case: a frontal chest radiograph (left panel) and three short reports A, B and C for the finding named under the radiograph. Reports A, B and C each show a sentence and a coloured overlay on the radiograph; brighter colour means more weight on that region. Rate only what is visible in the image. This is a research rating of automatically generated text, not clinical advice or a diagnosis; give your best judgement for every case and do not decline.

For each case give:
finding_present  yes, no or uncertain: is the named finding visible on the radiograph? Judge the radiograph alone, not the overlays or sentences.
sentence_A, sentence_B, sentence_C  0 = the sentence names a different finding from the one asked about, or the wrong side or zone; 1 = the finding is right and the location is missing or only partly right; 2 = the finding and its location are right. Use patient left and right (patient left is image right). If you judged the finding absent, use 0.
highlight_A, highlight_B, highlight_C  0 = the bright region is not on the finding; 1 = partly on it; 2 = on it. If you judged the finding absent, use 0.
preferred  A, B, C or none: the report whose sentence and overlay together you would rather receive.

Output rules: reply with CSV only, no other text, no code fence. First line exactly:
case,finding_present,sentence_A,highlight_A,sentence_B,highlight_B,sentence_C,highlight_C,preferred
then one line per case, in the order of the file names, using the case id from the file name (for example case_017). Use only the allowed values."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, default=["runs/main", "runs/plain_vlm"], help="guided and unguided run directories")
    ap.add_argument("--template_run", default="runs/classifier_only")
    ap.add_argument("--sets_yaml", required=True)
    ap.add_argument("--n_per_set", type=int, default=24)
    ap.add_argument("--controls", type=int, default=18, help="mismatched cases: the radiograph of one case with the sentences and overlays of another case with a different finding")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="adjudication")
    ap.add_argument("--select_only", action="store_true")
    a = ap.parse_args()
    ARMS[0], ARMS[1], ARMS[2] = ("guided", a.runs[0], "eval", "text"), ("unguided", a.runs[1], "eval", "text"), ("template", a.template_run, "eval_template_gated", "evidence")
    rng = np.random.default_rng(a.seed + 1)
    real = select(None, a.n_per_set, a.seed)
    if not real:
        raise SystemExit("no cases found: the guided, unguided and template evaluations must exist for at least one corpus")
    for r, cs in enumerate(real):
        cs.update(base=r, donor=r, control=0)
    items = list(real)
    for b in rng.permutation(len(real)):
        if len(items) - len(real) >= a.controls:
            break
        donors = [d for d in range(len(real)) if real[d]["set"] == real[b]["set"] and real[d]["cls"] != real[b]["cls"]
                  and not {real[d]["cls"], real[b]["cls"]} <= OPACITY]
        if donors:
            d = int(rng.choice(donors))
            items.append(dict(set=real[b]["set"], idx=real[b]["idx"], cls=real[b]["cls"], base=int(b), donor=d, control=1,
                              **{f"text_{n}": real[d][f"text_{n}"] for n, *_ in ARMS}))
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    for j, cs in enumerate(items):
        cs["case"] = f"case_{j + 1:03d}"
        cs["perm"] = [ARMS[k][0] for k in rng.permutation(3)]
    os.makedirs(os.path.join(a.out, "cases"), exist_ok=True)
    key = []
    for cs in items:
        key.append(dict(case=cs["case"], set=cs["set"], idx=cs["idx"], finding=cs["cls"], control=cs["control"], donor_idx=real[cs["donor"]]["idx"], donor_finding=real[cs["donor"]]["cls"],
                        **{f"arm_{L}": cs["perm"][k] for k, L in enumerate("ABC")}, **{f"text_{n}": cs[f"text_{n}"] for n, *_ in ARMS}))
    pd.DataFrame(key).to_csv(os.path.join(a.out, "key.csv"), index=False)
    cols = ["case", "finding", "reader", "finding_present"] + [f"{m}_{L}" for L in "ABC" for m in ("sentence", "highlight")] + ["preferred", "comment"]
    pd.DataFrame([{**{c: "" for c in cols}, "case": cs["case"], "finding": cs["cls"].replace("_", " ")} for cs in items])[cols].to_csv(
        os.path.join(a.out, "ratings_template.csv"), index=False)
    open(os.path.join(a.out, "instructions.txt"), "w").write(INSTRUCTIONS)
    print(f"{len(items)} cases ({len(items) - len(real)} mismatched controls); key, instructions and template written to {a.out}")
    if a.select_only:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evialign.metrics import upsample_map
    M = {}
    for name, run, sub, which in ARMS:
        M[name] = maps(run, [dict(c, sentence=c[f"text_{name}"]) for c in real], a.sets_yaml, which)
    for cs in items:
        fig, ax = plt.subplots(1, 4, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.2))
        img = M["guided"][cs["base"]][0]
        ax[0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[0].set_title("Radiograph", fontsize=9)
        ax[0].text(0.0, -0.04, f"Finding asked about: {cs['cls'].replace('_', ' ')}", transform=ax[0].transAxes, fontsize=8, va="top")
        for k, L in enumerate("ABC"):
            name = cs["perm"][k]
            mp = np.clip(upsample_map(M[name][cs["donor"]][1], img.shape[0]), 0, None)
            ax[k + 1].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[k + 1].imshow(mp / max(mp.max(), 1e-8), cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            ax[k + 1].set_title(f"Report {L}", fontsize=9)
            ax[k + 1].text(0.0, -0.04, "\n".join(textwrap.wrap(cs[f"text_{name}"], 40)[:4]), transform=ax[k + 1].transAxes, fontsize=8, va="top")
        for x in ax:
            x.set_xticks([]); x.set_yticks([])
        fig.savefig(os.path.join(a.out, "cases", cs["case"] + ".png"), dpi=130)
        plt.close(fig)
    # batches for readers that accept a limited number of images per message
    import shutil
    ids = [cs["case"] for cs in items]
    for b in range(0, len(ids), a.batch_size):
        d = os.path.join(a.out, "batches", f"batch_{b // a.batch_size + 1:02d}")
        os.makedirs(d, exist_ok=True)
        for c in ids[b:b + a.batch_size]:
            shutil.copy(os.path.join(a.out, "cases", c + ".png"), d)
    open(os.path.join(a.out, "panel_prompt.txt"), "w").write(PANEL_PROMPT)
    print("cases rendered; batches written")


if __name__ == "__main__":
    main()
