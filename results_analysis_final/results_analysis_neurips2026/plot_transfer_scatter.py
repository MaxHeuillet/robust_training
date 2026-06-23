#!/usr/bin/env python3
"""
plot_transfer_scatter.py
x: clean accuracy  |  y: avg robust accuracy (L1@300, L2@8, Linf@30, sev@3)
color: target model  |  shape: surrogate family
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from pathlib import Path

EXPERIMENT_META = {
    "test_v1":                    ("__clean__",       "__clean__"),
    "adv_clip_l1_eps75":          ("CLIP B/16",       r"$\ell_1$@75"),
    "adv_clip_l1_eps300":         ("CLIP B/16",       r"$\ell_1$@300"),
    "adv_clip_l2_eps2":           ("CLIP B/16",       r"$\ell_2$@2"),
    "adv_clip_l2_eps8":           ("CLIP B/16",       r"$\ell_2$@8"),
    "adv_clip_linf8":             ("CLIP B/16",       r"$\ell_\infty$@8"),
    "adv_linf30":                 ("CLIP B/16",       r"$\ell_\infty$@30"),
    "adv_clip_vith14_l1_eps75":   ("CLIP H/14",       r"$\ell_1$@75"),
    "adv_clip_vith14_l1_eps300":  ("CLIP H/14",       r"$\ell_1$@300"),
    "adv_clip_vith14_l2_eps2":    ("CLIP H/14",       r"$\ell_2$@2"),
    "adv_clip_vith14_l2_eps8":    ("CLIP H/14",       r"$\ell_2$@8"),
    "adv_clip_vith14_linf8":      ("CLIP H/14",       r"$\ell_\infty$@8"),
    "adv_clip_vith14_linf30":     ("CLIP H/14",       r"$\ell_\infty$@30"),
    "adv_metaclip_l1_eps75":      ("MetaCLIP H/14",   r"$\ell_1$@75"),
    "adv_metaclip_l1_eps300":     ("MetaCLIP H/14",   r"$\ell_1$@300"),
    "adv_metaclip_l2_eps2":       ("MetaCLIP H/14",   r"$\ell_2$@2"),
    "adv_metaclip_l2_eps8":       ("MetaCLIP H/14",   r"$\ell_2$@8"),
    "adv_metaclip_linf_eps8":     ("MetaCLIP H/14",   r"$\ell_\infty$@8"),
    "adv_metaclip_linf_eps30":    ("MetaCLIP H/14",   r"$\ell_\infty$@30"),
    "adv_siglip2_linf8":          ("SigLIP2 base",    r"$\ell_\infty$@8"),
    "adv_siglip2_linf30":         ("SigLIP2 base",    r"$\ell_\infty$@30"),
    "adv_siglip2_384_l1_eps75":   ("SigLIP2 SO400M",  r"$\ell_1$@75"),
    "adv_siglip2_384_l1_eps300":  ("SigLIP2 SO400M",  r"$\ell_1$@300"),
    "adv_siglip2_384_l2_eps2":    ("SigLIP2 SO400M",  r"$\ell_2$@2"),
    "adv_siglip2_384_l2_eps8":    ("SigLIP2 SO400M",  r"$\ell_2$@8"),
    "adv_siglip2_384_linf_eps8":  ("SigLIP2 SO400M",  r"$\ell_\infty$@8"),
    "adv_siglip2_384_linf_eps30": ("SigLIP2 SO400M",  r"$\ell_\infty$@30"),
    "adv_siglip2_correct_naflex_linf_eps30": ("SigLIP2 NaFlex", r"$\ell_\infty$@30"),
    "common_severity3":           ("__common__",      "sev@3"),
}

ROBUST_CONDITIONS = {r"$\ell_1$@300", r"$\ell_2$@8", r"$\ell_\infty$@30",} # "sev@3"

DATASETS = [
    "caltech101", "fgvc-aircraft-2013b", "flowers-102",
    "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset",
]

TARGETS = {"google_nothink": "Gemini Flash (no think)", "openai": "GPT-4o"}

TARGET_COLORS  = {"google_nothink": "#3DBD8A", "openai": "#5B9BD5"}

def surrogate_family(s):
    if s.startswith("CLIP"):     return "CLIP"
    if s.startswith("MetaCLIP"): return "MetaCLIP"
    if s.startswith("SigLIP"):   return "SigLIP2"
    return "Other"

FAMILY_MARKERS = {"CLIP": "o", "MetaCLIP": "D", "SigLIP2": "*", "Other": "s"}
FAMILY_MS      = {"CLIP": 8,   "MetaCLIP": 8,   "SigLIP2": 11,  "Other": 7}

SURROGATE_SHORT = {
    "CLIP B/16":       "CLIP B/16",
    "CLIP H/14":       "CLIP H/14",
    "MetaCLIP H/14":   "MetaCLIP H/14",
    "SigLIP2 base":    "SiG2 base",
    "SigLIP2 SO400M":  "SiG2 SO400M",
    "SigLIP2 NaFlex":  "SiG2 NaFlex",
}

# Manual label offsets (in axes-fraction points) to avoid overlap
# (dx_pts, dy_pts, ha)  — tuned to the zoomed window
LABEL_OFFSETS = {
    "CLIP B/16":      ( 6,   2, "left"),
    "CLIP H/14":      ( 6,  -8, "left"),
    "MetaCLIP H/14":  ( 6,   2, "left"),
    "SigLIP2 base":   (-6,   4, "right"),
    "SigLIP2 SO400M": (-6,  -8, "right"),
    "SigLIP2 NaFlex": ( 6,   2, "left"),
}

BASE = Path("llm_classification_results")

def dataset_key(run_name):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(ds):
            return ds
    return run_name.split("__")[0]

def load_predictions(run_name):
    p = BASE / run_name / "predictions.jsonl"
    if not p.exists():
        return None
    correct = total = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
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
    total   = len(seen)
    correct = sum(1 for r in seen.values() if r.get("correct", False))
    return (correct, total) if total > 0 else None

# ── Load ──────────────────────────────────────────────────────────────────────
acc: dict[tuple, float] = {}
for mp in sorted(BASE.glob("batch_manifest__all_datasets__*.json")):
    exp = mp.stem.replace("batch_manifest__all_datasets__", "")
    if exp not in EXPERIMENT_META: continue
    surrogate, pert = EXPERIMENT_META[exp]
    manifest = json.loads(mp.read_text())
    groups: dict[tuple, list] = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)
    for (ds, key), entries in groups.items():
        if key not in TARGETS: continue
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
            acc[(surrogate, pert, key, ds)] = correct / total

def mean_ds(surrogate, pert, target):
    vals = [acc.get((surrogate, pert, target, ds), np.nan) for ds in DATASETS]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals) if vals else np.nan

all_surrogates = sorted({s for (s, p, t, d) in acc if not s.startswith("__")})

points = []
for surrogate in all_surrogates:
    for tgt_key in TARGETS:
        clean_vals = [acc.get(("__clean__", "__clean__", tgt_key, ds), np.nan) for ds in DATASETS]
        clean_vals = [v for v in clean_vals if not np.isnan(v)]
        clean_acc  = np.mean(clean_vals) if clean_vals else np.nan

        robust_per_cond = []
        for pert in ROBUST_CONDITIONS:
            src = "__common__" if pert == "sev@3" else surrogate
            v = mean_ds(src, pert, tgt_key)
            if not np.isnan(v): robust_per_cond.append(v)
        robust_acc = np.mean(robust_per_cond) if robust_per_cond else np.nan

        if np.isnan(clean_acc) and np.isnan(robust_acc): continue
        points.append({
            "surrogate":  surrogate,
            "target":     tgt_key,
            "clean_acc":  clean_acc,
            "robust_acc": robust_acc,
            "family":     surrogate_family(surrogate),
        })

# ── Plot ──────────────────────────────────────────────────────────────────────
NEURIPS_FONTSIZE = 7
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       NEURIPS_FONTSIZE,
    "axes.labelsize":  NEURIPS_FONTSIZE,
    "xtick.labelsize": NEURIPS_FONTSIZE - 1,
    "ytick.labelsize": NEURIPS_FONTSIZE - 1,
    "legend.fontsize": NEURIPS_FONTSIZE - 1,
    "figure.dpi":      300,
})

fig, ax = plt.subplots(figsize=(3.25, 3.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F7F7F7")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", length=2)
ax.grid(True, color="white", linewidth=0.8, zorder=0)

seen_families, seen_targets = set(), set()

# ── First pass: connecting lines (drawn below points) ────────────────────────
surr_pts = {}
for pt in points:
    if np.isnan(pt["clean_acc"]) or np.isnan(pt["robust_acc"]): continue
    surr_pts.setdefault(pt["surrogate"], {})[pt["target"]] = pt

for surr, tgt_dict in surr_pts.items():
    if "google_nothink" not in tgt_dict or "openai" not in tgt_dict: continue
    p1 = tgt_dict["google_nothink"]
    p2 = tgt_dict["openai"]
    ax.plot(
        [p1["clean_acc"], p2["clean_acc"]],
        [p1["robust_acc"], p2["robust_acc"]],
        color="#BBBBBB", linewidth=0.9, zorder=2, solid_capstyle="round",
    )

# ── Second pass: scatter points ───────────────────────────────────────────────
for pt in points:
    if np.isnan(pt["clean_acc"]) or np.isnan(pt["robust_acc"]): continue
    family = pt["family"]
    ms = FAMILY_MS[family]
    ax.scatter(
        pt["clean_acc"], pt["robust_acc"],
        s=ms**2,
        c=TARGET_COLORS[pt["target"]],
        marker=FAMILY_MARKERS[family],
        edgecolors="white", linewidths=0.5,
        alpha=0.95, zorder=4,
    )
    seen_families.add(family)
    seen_targets.add(pt["target"])

# ── Third pass: one label per surrogate at Gemini point, with slope ───────────
for surr, tgt_dict in surr_pts.items():
    # Anchor label on the Gemini point (higher robust acc → less crowded top)
    anchor_key = "google_nothink" if "google_nothink" in tgt_dict else list(tgt_dict)[0]
    pt_anchor  = tgt_dict[anchor_key]

    # Slope annotation
    slope_str = ""
    if "google_nothink" in tgt_dict and "openai" in tgt_dict:
        p1 = tgt_dict["google_nothink"]
        p2 = tgt_dict["openai"]
        dx = p2["clean_acc"]  - p1["clean_acc"]
        dy = p2["robust_acc"] - p1["robust_acc"]
        slope = dy / dx if abs(dx) > 1e-6 else np.nan
        if not np.isnan(slope):
            slope_str = f"\nslope {slope:+.1f}"

    short = SURROGATE_SHORT.get(surr, surr)
    dx_pts, dy_pts, ha = LABEL_OFFSETS.get(surr, (6, 2, "left"))

    ax.annotate(
        f"{short}{slope_str}",
        (pt_anchor["clean_acc"], pt_anchor["robust_acc"]),
        xytext=(dx_pts, dy_pts), textcoords="offset points",
        fontsize=NEURIPS_FONTSIZE - 2,
        color="#222222",
        ha=ha, va="center",
        linespacing=1.2,
        zorder=5,
    )

ax.set_xlabel("Avg. clean accuracy", labelpad=3)
ax.set_ylabel(
    r"Avg. robust accuracy" "\n"
    r"($\ell_1$@300, $\ell_2$@8, $\ell_\infty$@30, sev@3)",
    labelpad=3,
)
# ax.set_xlim(0.725, 0.955)
# ax.set_ylim(0.50,  0.90)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

# ── Legends ───────────────────────────────────────────────────────────────────
target_handles = [
    mpatches.Patch(color=TARGET_COLORS[k], label=TARGETS[k], alpha=0.88)
    for k in ["google_nothink", "openai"] if k in seen_targets
]
leg1 = ax.legend(
    handles=target_handles,
    title="Target model",
    title_fontsize=NEURIPS_FONTSIZE - 1,
    loc="lower right",
    frameon=True, framealpha=0.92, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.5, labelspacing=0.3,
)
ax.add_artist(leg1)

family_handles = []
for f in ["CLIP", "MetaCLIP", "SigLIP2"]:
    if f not in seen_families: continue
    family_handles.append(
        mlines.Line2D([], [], color="#666666",
                      marker=FAMILY_MARKERS[f], linestyle="None",
                      markersize=FAMILY_MS[f] * 0.75, label=f)
    )
ax.legend(
    handles=family_handles,
    title="Surrogate family",
    title_fontsize=NEURIPS_FONTSIZE - 1,
    loc="upper left",
    frameon=True, framealpha=0.92, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.5, labelspacing=0.3,
)

plt.tight_layout(pad=0.5)

out = Path("./results_analysis_neurips2026/transfer_pca.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()