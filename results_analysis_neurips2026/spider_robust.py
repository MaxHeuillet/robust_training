#!/usr/bin/env python3
"""
Spider / radar chart comparing robust generalisation across threat models.
Run from: robust_training/
  python results_analysis_neurips2026/spider_robust.py
"""

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR       = Path("./results_analysis_neurips2026")
CONVNEXT_EVAL_DIR = Path("./robustgenbench_eval/convnext_base_fb_in22k_TRADES_v2")

# ── Load data ─────────────────────────────────────────────────────────────────
tbl = pd.read_csv(RESULTS_DIR / "transfer_table.csv")
ft  = pd.read_csv(RESULTS_DIR / "fft50.csv", index_col=0)

DATASETS = sorted(tbl["dataset"].unique())
DATASET_NAME_MAP = {
    "Caltech-101":   "caltech101",
    "FGVC Aircraft": "fgvc-aircraft-2013b",
    "Flowers-102":   "flowers-102",
    "Oxford Pet":    "oxford-iiit-pet",
    "Stanford Cars": "stanford_cars",
    "UC Merced":     "uc-merced-land-use-dataset",
}

AXES = [
    (r"$\ell_1$@75",         "l1@75",  "L1_acc",    "l1_eps75"),
    (r"$\ell_2$@2",          "l2@2",   "L2_acc",    "l2_eps2"),
    (r"$\ell_\infty$@4/255", "linf@4", "Linf_acc",  "linf_eps4"),
    (r"Common",              "sev@3",  "common_acc", "common_severity3"),
    (r"Clean",               "clean",  "clean_acc",  "clean"),
]
N_AXES    = len(AXES)
ax_labels = [a[0] for a in AXES]

# ── Helpers ───────────────────────────────────────────────────────────────────
def llm_worst_sum(target, pert_col):
    if pert_col == "clean":
        sub  = tbl[(tbl["target"] == target) & (tbl["surrogate"] == "CLIP ViT-B/16")]
        vals = sub.set_index("dataset")["clean"]
    else:
        sub  = tbl[tbl["target"] == target]
        vals = sub.groupby("dataset")[pert_col].min()
    return float(sum(vals.get(ds, np.nan) for ds in DATASETS))

_ft_single = ft[(ft["backbone_name"] == "convnext_b,sup,in22k") & (ft["loss_function"] == "TRADES")]

def convnext_wb_sum(ft_col):
    return float(_ft_single[ft_col].sum() * 100)

def oracle_sum(ft_col):
    return float(ft.groupby("dataset")[ft_col].max().sum() * 100)

def _load_convnext_transfer():
    data = {}
    if not CONVNEXT_EVAL_DIR.exists():
        print(f"WARNING: {CONVNEXT_EVAL_DIR} not found.")
        return data
    for csv_path in CONVNEXT_EVAL_DIR.glob("*.csv"):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                data[(row["dataset"], row["label"])] = float(row["accuracy"])
    return data

_convnext_transfer = _load_convnext_transfer()
print(tbl[tbl["target"] == "Gemini Flash (no think)"][["dataset", "surrogate", "linf@4"]].to_string())



_SURROGATES = [
    "zeroshot_clip_vitb16_laion2b", "zeroshot_clip_vith14_laion2b",
    "zeroshot_metaclip_vith14_fullcc2_5b", "zeroshot_siglip2_so400m_patch14_384",
]

print("\nConvNeXt transfer coverage (linf_eps4):")
for ds in DATASETS:
    ds_key = DATASET_NAME_MAP[ds]
    print(f"\n  {ds}:")
    for surrogate in _SURROGATES:
        matching = [(lbl, round(v,4)) for (d, lbl), v in _convnext_transfer.items()
                    if d == ds_key and surrogate in lbl and "linf_eps4" in lbl]
        print(f"    {surrogate}: {matching}")

def convnext_transfer_worst_sum(label_fragment):
    total = 0.0
    for ds in DATASETS:
        ds_key = DATASET_NAME_MAP[ds]
        if label_fragment == "clean":
            acc = _convnext_transfer.get((ds_key, "clean"), np.nan)
        elif label_fragment == "common_severity3":
            acc = _convnext_transfer.get((ds_key, "common__common_severity3"), np.nan)
        else:
            accs = []
            for surrogate in _SURROGATES:
                accs.extend(v for (d, lbl), v in _convnext_transfer.items()
                            if d == ds_key and surrogate in lbl and label_fragment in lbl)
            acc = min(accs) if accs else np.nan
        total += acc if not np.isnan(acc) else 0.0
    return total * 100

# ── Assemble values ───────────────────────────────────────────────────────────
VMAX = 600.0

# Each entry: (label, color, linestyle, marker, values)
MODELS = [
    # Black-box
    ("Gemini Flash", "#2ECC96", "-",  "o", [llm_worst_sum("Gemini Flash (no think)", p) for _,p,_,_ in AXES]),
    ("GPT-4o",       "#5B9BD5", "-",  "o", [llm_worst_sum("GPT-4o", p)                  for _,p,_,_ in AXES]),
    ("ConvNeXt-B",   "#E05759", "-",  "s", [convnext_transfer_worst_sum(frag)            for _,_,_,frag in AXES]),
    # White-box
    ("ConvNeXt-B",   "#C0392B", "--", "s", [convnext_wb_sum(f)  for _,_,f,_ in AXES]),
    ("Oracle",       "#F28E2B", "--", "D", [oracle_sum(f)        for _,_,f,_ in AXES]),
]

print("Sum scores (max=600):")
for label, *_, vals in MODELS:
    print(f"  {label:<15} " + "  ".join(f"{v:6.1f}" for v in vals))

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   5.5,
    "figure.dpi":  300,
})

angles  = np.linspace(0, 2 * np.pi, N_AXES, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(1.75, 2.10), subplot_kw={"polar": True})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for label, color, ls, mk, vals in MODELS:
    norm   = [v / VMAX for v in vals]
    closed = norm + [norm[0]]
    alpha  = 0.06 if ls == "--" else 0.10
    ax.plot(angles, closed, color=color, linewidth=0.9, linestyle=ls, zorder=4)
    ax.fill(angles, closed, color=color, alpha=alpha, zorder=3)
    ax.scatter(angles[:-1], norm, s=7, marker=mk, color=color,
               edgecolors="white", linewidths=0.3, zorder=5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(ax_labels, fontsize=5.0, linespacing=1.1)
ax.tick_params(axis='x', pad=-4)

r_ticks = [300, 450, 600]
ax.set_yticks([v / VMAX for v in r_ticks])
ax.set_yticklabels([str(v) for v in r_ticks], fontsize=4.5, color="#999999")
ax.set_ylim(0, 1.04)
ax.set_rlabel_position(40)

ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.4, linestyle="--")
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.4)
ax.spines["polar"].set_visible(False)
theta_ring = np.linspace(0, 2 * np.pi, 300)
ax.plot(theta_ring, np.full_like(theta_ring, 1.0 / 1.04),
        color="#CCCCCC", linewidth=0.6, zorder=20, clip_on=False)

# ── Legend — two separate legends stacked ─────────────────────────────────────
FS = 4.5

bb_handles = [
    mlines.Line2D([], [], color="#2ECC96", lw=1.1, ls="-",  marker="o", ms=2.5, label="Gemini Flash"),
    mlines.Line2D([], [], color="#5B9BD5", lw=1.1, ls="-",  marker="o", ms=2.5, label="GPT-4o mini"),
    mlines.Line2D([], [], color="#E05759", lw=1.1, ls="-",  marker="s", ms=2.5, label="ConvNeXt-B"),
]
wb_handles = [
    mlines.Line2D([], [], color="#C0392B", lw=1.1, ls="--", marker="s", ms=2.5, label="ConvNeXt-B"),
    mlines.Line2D([], [], color="#F28E2B", lw=1.1, ls="--", marker="D", ms=2.5, label="Oracle"),
]

leg1 = ax.legend(
    handles=bb_handles,
    title="Black-box:",
    title_fontsize=FS,
    loc="lower left",
    bbox_to_anchor=(0.02, -0.32),
    frameon=False,
    fontsize=FS,
    handlelength=1.2,
    handletextpad=0.3,
    labelspacing=0.2,
)
leg1._legend_box.align = "left"
leg1.get_title().set_fontweight("bold")   # ← add this
ax.add_artist(leg1)

leg2 = ax.legend(
    handles=wb_handles,
    title="White-box:",
    title_fontsize=FS,
    loc="lower right",
    bbox_to_anchor=(0.98, -0.32),
    frameon=False,
    fontsize=FS,
    handlelength=1.2,
    handletextpad=0.3,
    labelspacing=0.2,
)
leg2._legend_box.align = "left"
leg2.get_title().set_fontweight("bold")   # ← add this

# ── Save ──────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "spider_robust.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
print(f"Saved -> {out}")