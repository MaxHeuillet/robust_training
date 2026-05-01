import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
datasets = ["Caltech-101", "FGVC\nAircraft", "Flowers-102",
            "Oxford\nPet", "Stanford\nCars", "UC Merced"]

models = {
    "CLIP B/16":       [88.9, 29.2, 71.5, 91.7, 88.7, 68.1],
    "CLIP H/14":       [90.1, 47.6, 81.5, 94.4, 93.7, 74.5],
    "MetaCLIP H/14":   [89.0, 53.5, 84.5, 95.4, 87.9, 79.3],
    "SigLIP2 base":    [92.1, 13.7, 84.7, 15.1,  4.2, 73.8],
    "SigLIP2 SO400M":  [92.1, 23.8, 91.7, 14.0,  5.3, 87.6],
    "SO400M NaFlex":   [90.6, 27.2, 90.8, 15.8,  5.0, 81.9],
}

colors = {
    "CLIP B/16":      "#90CAF9",
    "CLIP H/14":      "#42A5F5",
    "MetaCLIP H/14":  "#1A237E",
    "SigLIP2 base":   "#CE93D8",
    "SigLIP2 SO400M": "#AB47BC",
    "SO400M NaFlex":  "#7B1FA2",
}

# ── NeurIPS sizing ────────────────────────────────────────────────────────────
# NeurIPS textwidth ≈ 5.5 in; 25% of linewidth ≈ 1.375 in wide.
# A half-column (one column of two) ≈ 3.25 in is the smallest readable figure.
# "25% of linewidth" likely means a quarter-page figure spanning half a column,
# so we target ~3.25 in wide × 2.6 in tall (golden ratio) which fits comfortably
# as a \includegraphics[width=0.5\columnwidth] figure.
# All fonts sized to remain legible at that physical size (≥ 6 pt after scaling).

NEURIPS_FONTSIZE = 7        # base font size (pt) — matches NeurIPS body ~10pt scaled
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        NEURIPS_FONTSIZE,
    "axes.titlesize":   NEURIPS_FONTSIZE,
    "axes.labelsize":   NEURIPS_FONTSIZE,
    "xtick.labelsize":  NEURIPS_FONTSIZE - 1,
    "ytick.labelsize":  NEURIPS_FONTSIZE - 1,
    "legend.fontsize":  NEURIPS_FONTSIZE - 1,
    "figure.dpi":       300,
})

fig, ax = plt.subplots(figsize=(3.25, 2.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F5F5F5")

# ── Bar geometry ──────────────────────────────────────────────────────────────
n_datasets = len(datasets)
n_models   = len(models)
bar_width  = 0.11
group_gap  = 0.18

x_centers = np.arange(n_datasets) * (n_models * bar_width + group_gap)

for i, (model_name, values) in enumerate(models.items()):
    offsets = x_centers + (i - n_models / 2 + 0.5) * bar_width
    ax.bar(offsets, values, width=bar_width,
           color=colors[model_name], label=model_name, zorder=3)

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x_centers)
ax.set_xticklabels(datasets, linespacing=0.9)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.set_ylabel("Embedding-based zero-shot accuracy (%)", labelpad=3)

ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(axis="both", length=0, pad=2)

# ── Legend — placed inside upper-left to save vertical space ─────────────────
handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
ax.legend(
    handles=handles,
    loc="upper right",
    ncol=3,
    frameon=True,
    framealpha=0.85,
    edgecolor="none",
    handlelength=0.9,
    handletextpad=0.4,
    columnspacing=0.6,
    borderpad=0.4,
    labelspacing=0.3,
)

plt.tight_layout(pad=0.4)
plt.savefig(
    "./results_analysis_neurips2026/vlm_performance.png",
    dpi=300,
    bbox_inches="tight",
)
print("Figure saved to ./results_analysis_neurips2026/vlm_performance.png")
plt.show()