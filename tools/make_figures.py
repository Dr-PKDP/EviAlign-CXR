"""Manuscript figures (SVG; black text, legends outside panels, compact layout).

python tools/make_figures.py --tables results/tables --main_run runs/main --sets_yaml configs/eval_sets.yaml --out results/figures
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams.update({"font.size": 8, "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black", "axes.edgecolor": "black", "svg.fonttype": "none", "font.family": "DejaVu Sans"})
PAL = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2e75b6", "#a50021", "#404040"]


def dot_panel(ax, df, metric, runs, sets, label):
    for k, r in enumerate(runs):
        sub = df[df["run"] == r].set_index("set")
        xs = np.arange(len(sets)) + (k - (len(runs) - 1) / 2) * 0.12
        ys = [sub[metric].get(s, np.nan) if metric in sub else np.nan for s in sets]
        ax.plot(xs, ys, "o", color=PAL[k % len(PAL)], ms=4, label=r)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, rotation=30, ha="right")
    ax.set_ylabel(label)
    ax.grid(axis="y", lw=0.3, color="#bbbbbb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def summary_figure(tables, out):
    df = pd.read_csv(os.path.join(tables, "all_metrics.csv"))
    runs = list(dict.fromkeys(df["run"]))
    sets = list(dict.fromkeys(df["set"]))
    metrics = [("macro_auroc", "Macro AUROC"), ("pec_macro_f1", "Prediction–explanation F1"),
               ("tgd_normalised", "Normalised text-guided deletion"), ("ecs_gap", "Counterfactual sensitivity gap")]
    metrics = [m for m in metrics if m[0] in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.9 * len(metrics) + 1.2, 2.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (m, lab) in zip(axes, metrics):
        dot_panel(ax, df, m, runs, sets, lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside right center", frameon=False)
    fig.savefig(os.path.join(out, "fig_summary.svg"))
    plt.close(fig)


def localisation_figure(tables, out, run):
    df = pd.read_csv(os.path.join(tables, "all_metrics.csv"))
    df = df[df["run"] == run]
    cols = [c for c in df.columns if c.startswith("loc_") and c.endswith("_hit")]
    if not cols:
        return
    maps = [c[4:-4] for c in cols]
    sets = [s for s in df["set"] if df.loc[df["set"] == s, cols].notna().any(axis=None)]
    fig, ax = plt.subplots(figsize=(3.6, 2.1), constrained_layout=True)
    w = 0.8 / len(maps)
    for k, (c, m) in enumerate(zip(cols, maps)):
        ys = [df.loc[df["set"] == s, c].values[0] for s in sets]
        ax.bar(np.arange(len(sets)) + (k - (len(maps) - 1) / 2) * w, ys, w, color=PAL[k % len(PAL)], label=m)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, rotation=30, ha="right")
    ax.set_ylabel("Pointing-game hit rate")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(loc="outside right center", frameon=False)
    fig.savefig(os.path.join(out, f"fig_localisation_{run}.svg"))
    plt.close(fig)


def qualitative_figure(run, sets_yaml, out, set_name, n=4, classes=None):
    from PIL import Image
    from evialign.data import ManifestDataset, letterbox
    from evialign.metrics import upsample_map
    p = os.path.join(run, "eval", f"maps_{set_name}.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    sets = {s["name"]: s for s in yaml.safe_load(open(sets_yaml))["sets"]}
    df = pd.read_csv(sets[set_name]["manifest"], low_memory=False)
    df = df[df["split"] == sets[set_name].get("split", "test")].reset_index(drop=True)
    per = pd.read_csv(os.path.join(run, "eval", f"per_image_{set_name}.csv"))
    k = min(n, len(z["idx"]))
    fig, axes = plt.subplots(k, 3, figsize=(4.6, 1.55 * k), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r in range(k):
        idx, c = int(z["idx"][r]), int(z["cls"][r])
        img = np.asarray(Image.open(df.loc[idx, "image_path"]).convert("L"))
        img, *_ = letterbox(img, 448)
        for j, key in enumerate(["evidence", "text", "gradcam"]):
            m = z[key][r]
            ax = axes[r, j]
            ax.imshow(img, cmap="gray")
            if not np.isnan(m).all():
                ax.imshow(upsample_map(np.nan_to_num(m), 448), cmap="inferno", alpha=0.45)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title({"evidence": "Classifier evidence", "text": "Explanation grounding", "gradcam": "Grad-CAM"}[key])
    fig.savefig(os.path.join(out, f"fig_qualitative_{set_name}.svg"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--main_run", required=True)
    ap.add_argument("--sets_yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qual_sets", nargs="*", default=["nih_test", "chestxdet"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary_figure(a.tables, a.out)
    localisation_figure(a.tables, a.out, os.path.basename(a.main_run.rstrip("/")))
    for s in a.qual_sets:
        qualitative_figure(a.main_run, a.sets_yaml, a.out, s)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
