#!/usr/bin/env python3
"""
Point plot with per-dataset dots + mean marker + range whisker.
Layout mirrors the grouped bar chart:
  super-groups = target model (Gemini | GPT-4o)
  x positions  = perturbation condition
  sub-groups   = surrogate  (color-coded, horizontally dodged)
Each surrogate at each perturbation shows:
  · individual dataset dots (small, semi-transparent)
  · mean dot (larger, opaque)
  · min–max whisker
Clean accuracy = horizontal dashed line per target super-group.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from pathlib import Path

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

PERT_ORDER  = ["l1@75","l1@300","l2@2","l2@8","linf@8","linf@30","sev@3"]
PERT_LABELS = {
    "l1@75":   r"$\ell_1$@75",  "l1@300":  r"$\ell_1$@300",
    "l2@2":    r"$\ell_2$@2",   "l2@8":    r"$\ell_2$@8",
    "linf@8":  r"$\ell_\infty$@8", "linf@30": r"$\ell_\infty$@30",
    "sev@3":   "sev@3",
}

SURROGATE_ORDER = ["CLIP B/16","CLIP H/14","MetaCLIP H/14", "SigLIP2 SO400M", ] #"SigLIP2 base", "SigLIP2 NaFlex"

TARGET_ORDER  = ["google_nothink", "openai"]
TARGET_LABELS = {"google_nothink": "Gemini Flash (no think)", "openai": "GPT-4o"}
TARGET_CLEAN_COLORS = {"google_nothink": "#1A9E6E", "openai": "#2E6FAD"}

# Surrogate palette: blue family → CLIP, green → MetaCLIP, red family → SigLIP2
SURROGATE_COLORS = {
    "CLIP B/16":       "#6BAED6",
    "CLIP H/14":       "#2171B5",
    "MetaCLIP H/14":   "#238B45",
    "SigLIP2 base":    "#FB6A4A",
    "SigLIP2 SO400M":  "#CB181D",
    "SigLIP2 NaFlex":  "#99000D",
}

DATASETS = ["caltech101","fgvc-aircraft-2013b","flowers-102",
            "oxford-iiit-pet","stanford_cars","uc-merced-land-use-dataset"]
BASE = Path("llm_classification_results")

# ── Load ──────────────────────────────────────────────────────────────────────
def dataset_key(rn):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if rn.startswith(ds): return ds
    return rn.split("__")[0]

def load_preds(run_name):
    p = BASE / run_name / "predictions.jsonl"
    if not p.exists(): return None
    c = t = 0
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            if r.get("error"): continue
            t += 1
            if r.get("correct", False): c += 1
        except: pass
    return (c, t) if t > 0 else None

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
    exp = mp.stem.replace("batch_manifest__all_datasets__","")
    if exp not in EXPERIMENT_META: continue
    surr, pert = EXPERIMENT_META[exp]
    manifest   = json.loads(mp.read_text())
    groups     = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)
    for (ds, key), entries in groups.items():
        if key not in {"google_nothink","openai"}: continue
        main_e  = [e for e in entries if "__complement" not in e["run_name"]]
        comp_e  = [e for e in entries if "__complement"     in e["run_name"]]
        main_ok = [e for e in main_e if e["status"] != "failed"]
        main    = main_ok[0] if main_ok else (main_e[0] if main_e else entries[0])
        comp    = comp_e[0] if comp_e else None
        if main["status"] != "retrieved": continue
        res = (merge_complement(main["run_name"], comp["run_name"])
               if comp and comp["status"] == "retrieved"
               else load_preds(main["run_name"]))
        if res: raw[(surr, pert, key, ds)] = res[0] / res[1]

# helper: per-dataset values for (surrogate, pert, target)
def get_ds_vals(surr, pert, tgt):
    src = "__common__" if (pert == "sev@3" or surr == "__any__") else surr
    return [raw[(src, pert, tgt, ds)]
            for ds in DATASETS if (src, pert, tgt, ds) in raw]

# clean per target (mean over datasets)
clean = {}
for tgt in TARGET_ORDER:
    vals = [raw[("__clean__","__clean__", tgt, ds)]
            for ds in DATASETS if ("__clean__","__clean__", tgt, ds) in raw]
    clean[tgt] = float(np.mean(vals)) if vals else np.nan

# ── Layout ────────────────────────────────────────────────────────────────────
n_surr   = len(SURROGATE_ORDER)
n_pert   = len(PERT_ORDER)
dodge_w  = 0.10          # horizontal spread per surrogate
pert_gap = 0.30          # gap between pert conditions
tgt_gap  = 0.70          # gap between the two target super-groups

pert_step  = n_surr * dodge_w + pert_gap
surr_offsets = np.linspace(-(n_surr-1)/2*dodge_w,
                            (n_surr-1)/2*dodge_w, n_surr)

target_x_starts = []
cursor = 0.0
for ti in range(len(TARGET_ORDER)):
    target_x_starts.append(cursor)
    cursor += n_pert * pert_step + tgt_gap

def x_pos(ti, pi, si):
    return target_x_starts[ti] + pi * pert_step + surr_offsets[si]

# ── Plot ──────────────────────────────────────────────────────────────────────
FS = 7
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": FS,
    "axes.labelsize": FS, "xtick.labelsize": FS-1,
    "ytick.labelsize": FS-1, "legend.fontsize": FS-1,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(6.75, 2.9))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(left=False, bottom=False, colors="#444")
ax.yaxis.grid(True, color="white", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

xtick_pos, xtick_lab = [], []
target_span = []   # (x_left, x_right, tgt) for clean lines + labels

for ti, tgt in enumerate(TARGET_ORDER):
    x_left  = x_pos(ti, 0,            0) - dodge_w * 0.8
    x_right = x_pos(ti, n_pert - 1, n_surr - 1) + dodge_w * 0.8
    target_span.append((x_left, x_right, tgt))

    for pi, pert in enumerate(PERT_ORDER):
        pert_center = target_x_starts[ti] + pi * pert_step

        # light column shading — alternate
        if pi % 2 == 0:
            col_l = x_pos(ti, pi, 0) - dodge_w * 0.7
            col_r = x_pos(ti, pi, n_surr-1) + dodge_w * 0.7
            ax.axvspan(col_l, col_r, color="#EEEEEE", zorder=0, linewidth=0)

        xtick_pos.append(pert_center)
        xtick_lab.append(PERT_LABELS[pert])

        # ── sev@3: single surrogate-independent column ──────────────────
        if pert == "sev@3":
            SEV_COLOR = "#6B4C9A"   # distinct purple — not tied to any surrogate
            xc  = pert_center        # centred on the pert tick, no dodging
            vals = get_ds_vals("__any__", pert, tgt)   # surrogate-agnostic
            if vals:
                vmin, vmax = min(vals), max(vals)
                vmean      = float(np.mean(vals))
                ax.plot([xc, xc], [vmin, vmax],
                        color=SEV_COLOR, linewidth=1.2, alpha=0.65,
                        solid_capstyle="round", zorder=3)
                for cap_y in [vmin, vmax]:
                    ax.plot([xc - dodge_w*0.35, xc + dodge_w*0.35],
                            [cap_y, cap_y],
                            color=SEV_COLOR, linewidth=0.9, alpha=0.65, zorder=3)
                jitter = np.linspace(-dodge_w*0.22, dodge_w*0.22, len(vals))
                for jx, v in zip(jitter, sorted(vals)):
                    ax.scatter(xc + jx, v, s=6, color=SEV_COLOR,
                               alpha=0.55, zorder=4, linewidths=0)
                ax.scatter(xc, vmean, s=28, color=SEV_COLOR,
                           edgecolors="white", linewidths=0.9,
                           zorder=5, alpha=0.95)
            continue   # skip per-surrogate loop for sev@3

        # ── adversarial: one column per surrogate ────────────────────────────
        for si, surr in enumerate(SURROGATE_ORDER):
            xc    = x_pos(ti, pi, si)
            color = SURROGATE_COLORS[surr]
            vals  = get_ds_vals(surr, pert, tgt)
            if not vals:
                # tiny grey cross = no data
                ax.scatter(xc, 0.15, s=6, marker="x",
                           color="#BBBBBB", zorder=3, linewidths=0.6)
                continue

            vmin, vmax = min(vals), max(vals)
            vmean      = float(np.mean(vals))

            # whisker (min–max)
            ax.plot([xc, xc], [vmin, vmax],
                    color=color, linewidth=1.0, alpha=0.55,
                    solid_capstyle="round", zorder=3)
            # caps
            for cap_y in [vmin, vmax]:
                ax.plot([xc - dodge_w*0.25, xc + dodge_w*0.25],
                        [cap_y, cap_y],
                        color=color, linewidth=0.8, alpha=0.6, zorder=3)
            # individual dataset dots
            jitter = np.linspace(-dodge_w*0.18, dodge_w*0.18, len(vals))
            for jx, v in zip(jitter, sorted(vals)):
                ax.scatter(xc + jx, v, s=5, color=color,
                           alpha=0.50, zorder=4, linewidths=0)
            # mean marker — filled circle with white edge
            ax.scatter(xc, vmean, s=22, color=color,
                       edgecolors="white", linewidths=0.8,
                       zorder=5, alpha=0.95)

# ── Clean accuracy lines ──────────────────────────────────────────────────────
for x_left, x_right, tgt in target_span:
    y = clean[tgt]
    if np.isnan(y): continue
    ax.hlines(y, x_left, x_right,
              colors=TARGET_CLEAN_COLORS[tgt],
              linewidths=1.4, linestyles="--", zorder=6)
    ax.text(x_right + 0.05, y,
            f"clean\n{y*100:.1f}%",
            va="center", ha="left",
            fontsize=FS - 2, color=TARGET_CLEAN_COLORS[tgt],
            linespacing=1.1)

# ── Target super-group labels & divider ──────────────────────────────────────
for x_left, x_right, tgt in target_span:
    cx = (x_left + x_right) / 2
    ax.text(cx, -0.13, TARGET_LABELS[tgt],
            ha="center", va="top",
            fontsize=FS, fontweight="bold", color="#222",
            transform=ax.get_xaxis_transform())

if len(target_span) == 2:
    xd = (target_span[0][1] + target_span[1][0]) / 2
    ax.axvline(xd, color="#CCCCCC", linewidth=0.9,
               linestyle="--", zorder=1)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, rotation=35, ha="right")
ax.set_xlim(target_span[0][0] - 0.15,
            target_span[-1][1] + 0.55)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
ax.set_ylabel("Accuracy per dataset", labelpad=4)

# ── Legend ────────────────────────────────────────────────────────────────────
# Surrogate color swatches
surr_handles = [
    mpatches.Patch(color=SURROGATE_COLORS[s], label=s, alpha=0.88)
    for s in SURROGATE_ORDER
] + [mpatches.Patch(color="#6B4C9A", label="(common corruptions)", alpha=0.88)]
# Anatomy guide
anatomy = [
    mlines.Line2D([],[],color="#777",linewidth=1.0,label="min–max range"),
    mlines.Line2D([],[],marker="o",color="w",markerfacecolor="#777",
                  markeredgecolor="white",markersize=4,label="mean"),
    mlines.Line2D([],[],marker="o",color="w",markerfacecolor="#777",
                  markersize=2.5,alpha=0.5,label="dataset"),
]
leg1 = ax.legend(
    handles=surr_handles,
    title="Surrogate", title_fontsize=FS-1,
    ncol=3, loc="lower right",
    frameon=True, framealpha=0.95, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.5, labelspacing=0.25, columnspacing=0.8,
)
ax.add_artist(leg1)
ax.legend(
    handles=anatomy,
    title="Symbol key", title_fontsize=FS-1,
    loc="lower center",
    frameon=True, framealpha=0.95, edgecolor="none",
    handlelength=1.0, handletextpad=0.4,
    borderpad=0.5, labelspacing=0.3,
)

plt.tight_layout(pad=0.4)
plt.subplots_adjust(bottom=0.22)

out = Path("./results_analysis_neurips2026/transfer_pointplot.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()