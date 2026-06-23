#!/usr/bin/env python3
"""
Grouped bar chart: x = perturbation condition, groups = target model,
subgroups = surrogate. Horizontal line = clean accuracy per target.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_META = {
    "test_v1":                    ("__clean__",      "__clean__"),
    "adv_clip_l1_eps75":          ("CLIP B/16",      "l1@75"),
    "adv_clip_l1_eps300":         ("CLIP B/16",      "l1@300"),
    "adv_clip_l2_eps2":           ("CLIP B/16",      "l2@2"),
    "adv_clip_l2_eps8":           ("CLIP B/16",      "l2@8"),
    "adv_clip_linf8":             ("CLIP B/16",      "linf@8"),
    "adv_linf30":                 ("CLIP B/16",      "linf@30"),
    "adv_clip_vith14_l1_eps75":   ("CLIP H/14",      "l1@75"),
    "adv_clip_vith14_l1_eps300":  ("CLIP H/14",      "l1@300"),
    "adv_clip_vith14_l2_eps2":    ("CLIP H/14",      "l2@2"),
    "adv_clip_vith14_l2_eps8":    ("CLIP H/14",      "l2@8"),
    "adv_clip_vith14_linf8":      ("CLIP H/14",      "linf@8"),
    "adv_clip_vith14_linf30":     ("CLIP H/14",      "linf@30"),
    "adv_metaclip_l1_eps75":      ("MetaCLIP H/14",  "l1@75"),
    "adv_metaclip_l1_eps300":     ("MetaCLIP H/14",  "l1@300"),
    "adv_metaclip_l2_eps2":       ("MetaCLIP H/14",  "l2@2"),
    "adv_metaclip_l2_eps8":       ("MetaCLIP H/14",  "l2@8"),
    "adv_metaclip_linf_eps8":     ("MetaCLIP H/14",  "linf@8"),
    "adv_metaclip_linf_eps30":    ("MetaCLIP H/14",  "linf@30"),
    "adv_siglip2_linf8":          ("SigLIP2 base",   "linf@8"),
    "adv_siglip2_linf30":         ("SigLIP2 base",   "linf@30"),
    "adv_siglip2_384_l1_eps75":   ("SigLIP2 SO400M", "l1@75"),
    "adv_siglip2_384_l1_eps300":  ("SigLIP2 SO400M", "l1@300"),
    "adv_siglip2_384_l2_eps2":    ("SigLIP2 SO400M", "l2@2"),
    "adv_siglip2_384_l2_eps8":    ("SigLIP2 SO400M", "l2@8"),
    "adv_siglip2_384_linf_eps8":  ("SigLIP2 SO400M", "linf@8"),
    "adv_siglip2_384_linf_eps30": ("SigLIP2 SO400M", "linf@30"),
    "adv_siglip2_correct_naflex_linf_eps30": ("SigLIP2 NaFlex", "linf@30"),
    "common_severity3":           ("__common__",     "sev@3"),
}

# Perturbation order on x-axis (shared across both target groups)
PERT_ORDER = ["l1@75", "l1@300", "l2@2", "l2@8", "linf@8", "linf@30", "sev@3"]
PERT_LABELS = {
    "l1@75":    r"$\ell_1$@75",
    "l1@300":   r"$\ell_1$@300",
    "l2@2":     r"$\ell_2$@2",
    "l2@8":     r"$\ell_2$@8",
    "linf@8":   r"$\ell_\infty$@8",
    "linf@30":  r"$\ell_\infty$@30",
    "sev@3":    "sev@3",
}

# Surrogates in display order
SURROGATE_ORDER = [
    "CLIP B/16", "CLIP H/14", "MetaCLIP H/14", "SigLIP2 SO400M",
] #"SigLIP2 base", "SigLIP2 NaFlex",

# Target display order
TARGET_ORDER = ["google_nothink", "openai"]
TARGET_LABELS = {"google_nothink": "Gemini Flash (no think)", "openai": "GPT-4o"}
TARGET_CLEAN_COLORS = {"google_nothink": "#1A9E6E", "openai": "#2E6FAD"}

# Surrogate colors — family palette
SURROGATE_COLORS = {
    "CLIP B/16":       "#6BAED6",   # blue family
    "CLIP H/14":       "#2171B5",
    "MetaCLIP H/14":   "#238B45",   # green
    "SigLIP2 base":    "#FB6A4A",   # red/orange family
    "SigLIP2 SO400M":  "#CB181D",
    "SigLIP2 NaFlex":  "#99000D",
}

DATASETS = [
    "caltech101", "fgvc-aircraft-2013b", "flowers-102",
    "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset",
]
TARGETS_ALL = {"google_nothink", "openai"}
BASE = Path("llm_classification_results")

# ── Load ──────────────────────────────────────────────────────────────────────
def dataset_key(run_name):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(ds): return ds
    return run_name.split("__")[0]

def load_predictions(run_name):
    p = BASE / run_name / "predictions.jsonl"
    if not p.exists(): return None
    correct = total = 0
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:
            rec = json.loads(line)
            if rec.get("error"): continue
            total += 1
            if rec.get("correct", False): correct += 1
        except: pass
    return (correct, total) if total > 0 else None

def merge_complement(main_run, comp_run):
    recs = []
    for run in [main_run, comp_run]:
        p = BASE / run / "predictions.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try: recs.append(json.loads(line))
                    except: pass
    if not recs: return None
    seen = {}
    for r in recs:
        if not r.get("error") and r["index"] not in seen:
            seen[r["index"]] = r
    total = len(seen)
    correct = sum(1 for r in seen.values() if r.get("correct", False))
    return (correct, total) if total > 0 else None

raw: dict[tuple, float] = {}
for mp in sorted(BASE.glob("batch_manifest__all_datasets__*.json")):
    exp = mp.stem.replace("batch_manifest__all_datasets__", "")
    if exp not in EXPERIMENT_META: continue
    surrogate, pert = EXPERIMENT_META[exp]
    manifest = json.loads(mp.read_text())
    groups = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)
    for (ds, key), entries in groups.items():
        if key not in TARGETS_ALL: continue
        main_entries = [e for e in entries if "__complement" not in e["run_name"]]
        comp_entries = [e for e in entries if "__complement"     in e["run_name"]]
        main_ok = [e for e in main_entries if e["status"] != "failed"]
        main    = main_ok[0] if main_ok else (main_entries[0] if main_entries else entries[0])
        comp    = comp_entries[0] if comp_entries else None
        if main["status"] != "retrieved": continue
        result = (
            merge_complement(main["run_name"], comp["run_name"])
            if comp and comp["status"] == "retrieved"
            else load_predictions(main["run_name"])
        )
        if result:
            correct, total = result
            raw[(surrogate, pert, key, ds)] = correct / total

def mean_ds(surrogate, pert, target):
    vals = [raw.get((surrogate, pert, target, ds), np.nan) for ds in DATASETS]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan

# ── Build data matrix ─────────────────────────────────────────────────────────
# data[target][pert][surrogate] = mean acc over datasets
# clean[target] = clean acc
clean = {}
for tgt in TARGET_ORDER:
    vals = [raw.get(("__clean__", "__clean__", tgt, ds), np.nan) for ds in DATASETS]
    vals = [v for v in vals if not np.isnan(v)]
    clean[tgt] = float(np.mean(vals)) if vals else np.nan

data = {}
for tgt in TARGET_ORDER:
    data[tgt] = {}
    for pert in PERT_ORDER:
        data[tgt][pert] = {}
        for surr in SURROGATE_ORDER:
            src = "__common__" if pert == "sev@3" else surr
            data[tgt][pert][surr] = mean_ds(src, pert, tgt)

# ── Layout maths ──────────────────────────────────────────────────────────────
n_targets   = len(TARGET_ORDER)    # 2
n_perts     = len(PERT_ORDER)      # 7
n_surrogates = len(SURROGATE_ORDER) # 6

bar_w       = 0.11
surr_gap    = 0.02   # gap between bars within one (target, pert) group
pert_gap    = 0.18   # gap between pert conditions within one target group
target_gap  = 0.55   # big gap between the two target super-groups

group_width = n_surrogates * (bar_w + surr_gap) - surr_gap
pert_step   = group_width + pert_gap

# x positions of the leftmost bar for each (target_idx, pert_idx)
target_offsets = []
cursor = 0.0
for ti in range(n_targets):
    t_positions = []
    for pi in range(n_perts):
        t_positions.append(cursor)
        cursor += pert_step
    target_offsets.append(t_positions)
    cursor += target_gap   # big gap between target super-groups

# ── Plot ──────────────────────────────────────────────────────────────────────
FS = 7
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        FS,
    "axes.labelsize":   FS,
    "xtick.labelsize":  FS - 1,
    "ytick.labelsize":  FS - 1,
    "legend.fontsize":  FS - 1,
    "figure.dpi":       300,
})

fig, ax = plt.subplots(figsize=(6.75, 2.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(left=False, bottom=False, colors="#444")
ax.yaxis.grid(True, color="white", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

# ── Draw bars ─────────────────────────────────────────────────────────────────
xtick_pos, xtick_lab = [], []
target_center_x = []   # for super-group labels and clean lines

for ti, tgt in enumerate(TARGET_ORDER):
    t_positions = target_offsets[ti]
    pert_centers = []

    for pi, pert in enumerate(PERT_ORDER):
        x0 = t_positions[pi]
        bar_centers = []
        for si, surr in enumerate(SURROGATE_ORDER):
            xc = x0 + si * (bar_w + surr_gap)
            val = data[tgt][pert][surr]
            color = SURROGATE_COLORS[surr]
            if not np.isnan(val):
                ax.bar(xc, val, width=bar_w,
                       color=color, alpha=0.88,
                       zorder=3, linewidth=0)
            else:
                # hatched placeholder
                ax.bar(xc, 0.3, width=bar_w,
                       color="#DDDDDD", alpha=0.5,
                       zorder=3, linewidth=0, hatch="//")
            bar_centers.append(xc)

        center = np.mean(bar_centers)
        pert_centers.append(center)
        xtick_pos.append(center)
        xtick_lab.append(PERT_LABELS[pert])

    target_center_x.append((np.mean(pert_centers), tgt))

# ── Clean accuracy reference lines ────────────────────────────────────────────
for cx, tgt in target_center_x:
    # Span from first to last bar in this target group
    ti = TARGET_ORDER.index(tgt)
    x_left  = target_offsets[ti][0]
    x_right = target_offsets[ti][-1] + group_width
    y_clean = clean[tgt]
    if not np.isnan(y_clean):
        ax.hlines(y_clean, x_left - 0.05, x_right + 0.05,
                  colors=TARGET_CLEAN_COLORS[tgt],
                  linewidths=1.4, linestyles="-",
                  zorder=5)
        ax.text(x_right + 0.07, y_clean,
                f"clean\n{y_clean*100:.1f}%",
                va="center", ha="left",
                fontsize=FS - 2, color=TARGET_CLEAN_COLORS[tgt],
                linespacing=1.1)

# ── Target super-group labels & divider ───────────────────────────────────────
for cx, tgt in target_center_x:
    ax.text(cx, -0.115, TARGET_LABELS[tgt],
            ha="center", va="top",
            fontsize=FS, fontweight="bold",
            color="#222222",
            transform=ax.get_xaxis_transform())

# Vertical divider between the two target groups
if len(target_center_x) == 2:
    x_div = (target_offsets[0][-1] + group_width + target_offsets[1][0]) / 2
    ax.axvline(x_div, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, rotation=35, ha="right")
ax.set_xlim(target_offsets[0][0] - 0.15,
            target_offsets[-1][-1] + group_width + 0.45)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_ylabel("Avg. accuracy (over datasets)", labelpad=4)

# ── Surrogate legend ──────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=SURROGATE_COLORS[s], label=s, alpha=0.88)
    for s in SURROGATE_ORDER
]
ax.legend(
    handles=handles,
    title="Surrogate",
    title_fontsize=FS - 1,
    ncol=2,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True, framealpha=0.95, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.5, labelspacing=0.25, columnspacing=0.8,
)

plt.tight_layout(pad=0.4)
plt.subplots_adjust(bottom=0.22)

out = Path("./results_analysis_neurips2026/transfer_grouped_bar.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()