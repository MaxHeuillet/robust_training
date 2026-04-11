import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (from predictions.jsonl via summarize_results.py) ───────────────────
datasets = ["Caltech-101", "FGVC\nAircraft", "Flowers-102",
            "Oxford\nPet", "Stanford\nCars", "UC Merced"]

models = {
    "Gemini Flash\n(no think)": [93.0, 80.9, 95.7, 96.5, 93.9, 90.7],
    "GPT-4o":                   [90.6, 45.8, 73.7, 87.4, 72.2, 77.6],
}

colors = {
    "Gemini Flash\n(no think)": "#3DBD8A",   # teal-green
    "GPT-4o":                   "#5B9BD5",   # steel blue
}

# ── NeurIPS sizing (25 % of linewidth = 0.5 \columnwidth) ────────────────────
NEURIPS_FONTSIZE = 7
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
bar_width  = 0.25
group_gap  = 0.20

x_centers = np.arange(n_datasets) * (n_models * bar_width + group_gap)

for i, (model_name, values) in enumerate(models.items()):
    offsets = x_centers + (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(offsets, values, width=bar_width,
                  color=colors[model_name], label=model_name, zorder=3)
    # Value annotations on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{int(round(val))}",
            ha="center", va="bottom",
            fontsize=NEURIPS_FONTSIZE - 2,
            color="#333333",
        )

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x_centers)
ax.set_xticklabels(datasets, linespacing=0.9)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.set_ylabel("Accuracy (%)", labelpad=3)

ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(axis="both", length=0, pad=2)

# ── Legend — upper right, inside axes ────────────────────────────────────────
handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
ax.legend(
    handles=handles,
    loc="lower right",
    ncol=1,
    frameon=True,
    framealpha=0.85,
    edgecolor="none",
    handlelength=0.9,
    handletextpad=0.4,
    borderpad=0.4,
    labelspacing=0.3,
)

plt.tight_layout(pad=0.4)

import os
os.makedirs("./results_analysis_neurips2026", exist_ok=True)
plt.savefig(
    "./results_analysis_neurips2026/clean_llm_performance.png",
    dpi=300,
    bbox_inches="tight",
)
print("Figure saved to ./results_analysis_neurips2026/clean_llm_performance.png")
plt.show()