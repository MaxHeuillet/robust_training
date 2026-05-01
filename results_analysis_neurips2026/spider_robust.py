#!/usr/bin/env python3
"""
Spider / radar chart comparing robust generalisation:
  - Gemini Flash (no think)         worst-case over surrogates per (dataset, threat)
  - GPT-4o                          worst-case over surrogates per (dataset, threat)
  - ConvNeXt-B sup IN-22K (TRADES)  single best model, fixed across datasets
  - Neural oracle                   best model per (dataset, threat) independently

Axes  = L1@75, L2@2, Linf@4/255, Common (sev@3)
Metric = sum of accuracy over 6 datasets (max = 600)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────────────
tbl = pd.read_csv("./results_analysis_neurips2026/transfer_table.csv")
ft  = pd.read_csv("./results_analysis_neurips2026/fft50.csv", index_col=0)

DATASETS = sorted(tbl["dataset"].unique())   # 6 datasets

# Axes: (display label, tbl column, fft50 column)
AXES = [
    (r"$\ell_1$@75",         "l1@75",  "L1_acc"),
    (r"$\ell_2$@2",          "l2@2",   "L2_acc"),
    (r"$\ell_\infty$@4/255", "linf@4", "Linf_acc"),
    (r"Common",              "sev@3",  "common_acc"),
    (r"Clean",               "clean",  "clean_acc"),
]
N_AXES  = len(AXES)
ax_labels = [a[0] for a in AXES]

# ── LLM: worst-case sum (min over surrogates per dataset, then sum) ───────────
def llm_worst_sum(target, pert_col):
    if pert_col == "clean":
        # clean is surrogate-independent — take mean per dataset then sum
        sub = tbl[(tbl["target"] == target) & (tbl["surrogate"] == "CLIP ViT-B/16")]
        vals = sub.set_index("dataset")["clean"]
    else:
        sub   = tbl[tbl["target"] == target]
        vals  = sub.groupby("dataset")[pert_col].min()
    return float(sum(vals.get(ds, np.nan) for ds in DATASETS))

# ── FFT single model: convnext_b,sup,in22k TRADES ────────────────────────────
ft_single = ft[(ft["backbone_name"] == "convnext_b,sup,in22k") &
               (ft["loss_function"]  == "TRADES")]

def ft_single_sum(ft_col):
    return float(ft_single[ft_col].sum() * 100)

# ── Neural oracle: best model per (dataset, threat) independently ─────────────
def ft_oracle_sum(ft_col):
    return float(ft.groupby("dataset")[ft_col].max().sum() * 100)

# ── Assemble ──────────────────────────────────────────────────────────────────
VMAX = 600.0   # max possible (100% × 6 datasets)

models = {
    "Gemini Flash\n(no think)": [
        llm_worst_sum("Gemini Flash (no think)", p) for _, p, _ in AXES
    ],
    "GPT-4o": [
        llm_worst_sum("GPT-4o", p) for _, p, _ in AXES
    ],
    "ConvNeXt-B\nsup IN-22K\n(TRADES)": [
        ft_single_sum(f) for _, _, f in AXES
    ],
    "Neural oracle\n(best/dataset/threat)": [
        ft_oracle_sum(f) for _, _, f in AXES
    ],
}

print("Values (sum over 6 datasets, max=600):")
for name, vals in models.items():
    print(f"  {name.replace(chr(10),' '):40s}: "
          + "  ".join(f"{l.split('@')[0].strip()}={v:.1f}"
                      for (l,_,_), v in zip(AXES, vals)))

# ── NeurIPS sizing ────────────────────────────────────────────────────────────
# 25% of NeurIPS linewidth = 0.25 × 5.5in = 1.375in
# Radar charts need to be square; at this size axis labels must be very compact.
# We use 1.6 × 1.6 in (renders at ~1.375in after tight bbox trim of whitespace).
FIG_SIZE  = (1.75, 1.75)   # slightly generous; bbox_inches="tight" trims to ~1.4in
FS_BASE   = 5.5            # base font — matches ~8pt at 0.25 linewidth scaling
FS_TICK   = 4.5            # radial tick labels
FS_AXIS   = 5.0            # spoke (threat model) labels
FS_LEGEND = 4.5            # legend

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    FS_BASE,
    "figure.dpi":   300,
    "lines.linewidth": 0.9,
})

COLORS = {
    "Gemini Flash (no think)":         "#2ECC96",
    "GPT-4o":                          "#5B9BD5",
    "ConvNeXt-B sup IN-22K (TRADES)":  "#E05759",
    "Neural oracle":                   "#F28E2B",
}
# Shorter legend labels — no newlines needed at this size
LEGEND_LABELS = {
    "Gemini Flash (no think)":        "Gemini Flash (no think)",
    "GPT-4o":                         "GPT-4o",
    "ConvNeXt-B sup IN-22K (TRADES)": "ConvNeXt-B TRADES",
    "Neural oracle":                  "Neural oracle",
}
LINE_STYLES = {
    "Gemini Flash (no think)":        "-",
    "GPT-4o":                         "-",
    "ConvNeXt-B sup IN-22K (TRADES)": "-",
    "Neural oracle":                  "--",
}

# Remap model keys to short display keys
models_short = {
    "Gemini Flash (no think)":        models["Gemini Flash\n(no think)"],
    "GPT-4o":                         models["GPT-4o"],
    "ConvNeXt-B sup IN-22K (TRADES)": models["ConvNeXt-B\nsup IN-22K\n(TRADES)"],
    "Neural oracle":                  models["Neural oracle\n(best/dataset/threat)"],
}

angles = np.linspace(0, 2 * np.pi, N_AXES, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=FIG_SIZE, subplot_kw={"polar": True})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for name, vals in models_short.items():
    norm   = [v / VMAX for v in vals]
    closed = norm + [norm[0]]
    ls     = LINE_STYLES[name]
    mk     = "D" if ls == "--" else "o"

    ax.plot(angles, closed,
            color=COLORS[name], linewidth=0.9,
            linestyle=ls, zorder=4)
    ax.fill(angles, closed,
            color=COLORS[name],
            alpha=0.05 if ls == "--" else 0.10, zorder=3)
    ax.scatter(angles[:-1], norm,
               s=7, marker=mk, color=COLORS[name],
               edgecolors="white", linewidths=0.3, zorder=5)

# ── Spoke labels — pulled tight against the outermost ring ───────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(ax_labels, fontsize=FS_AXIS, linespacing=1.1)
# Negative pad pulls labels inside — right up against the ring
ax.tick_params(axis='x', pad=-4)

# ── Radial ticks ─────────────────────────────────────────────────────────────
r_ticks = [300, 450, 600]
ax.set_yticks([v / VMAX for v in r_ticks])
ax.set_yticklabels([str(v) for v in r_ticks],
                   fontsize=FS_TICK, color="#999999")
# 1.04 = just a sliver of room above max data; labels are now overlaid on ring
ax.set_ylim(0, 1.04)
ax.set_rlabel_position(40)

# ── Grid ─────────────────────────────────────────────────────────────────────
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.4, linestyle="--")
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.4)
# Hide default spine (it renders under tick labels) and redraw on top
ax.spines["polar"].set_visible(False)
theta_ring = np.linspace(0, 2 * np.pi, 300)
ax.plot(theta_ring, np.full_like(theta_ring, 1.0 / 1.04),
        color="#CCCCCC", linewidth=0.6,
        zorder=20, clip_on=False)

# ── Legend — inside lower portion of chart to eliminate external whitespace ───
handles = [
    mpatches.Patch(color=COLORS[n], alpha=0.88,
                   label=LEGEND_LABELS[n])
    for n in models_short
]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),   # tighter: was -0.38
    ncol=2,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=0.7,
    handletextpad=0.3,
    columnspacing=0.6,
    labelspacing=0.2,
)

plt.tight_layout(pad=0.0)
out = Path("./results_analysis_neurips2026/spider_robust.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()