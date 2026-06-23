#!/usr/bin/env python3
"""
plot_transfer_pca.py

Each point in the PCA = one (surrogate × target model) pair.
Feature vector = mean accuracy over datasets for each perturbation condition
  [L1-75, L1-300, L2-2, L2-8, Linf-8, Linf-30, common-sev3]

Visual encoding:
  color  → target model  (GPT-4o vs Gemini Flash no-think)
  shape  → surrogate family  (CLIP, MetaCLIP, SigLIP2)
  size   → mean accuracy across all conditions (larger = weaker transfer)
  label  → surrogate name

Save: ./results_analysis_neurips2026/transfer_pca.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from pathlib import Path
from sklearn.decomposition import PCA

# ── Experiment registry: stem → (surrogate_label, perturbation_condition) ────
EXPERIMENT_META = {
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
    "common_severity3":           ("Common corr.",    "sev@3"),
}

# Canonical perturbation axis order (= PCA feature dimensions)
PERT_ORDER = [
    r"$\ell_1$@75",
    r"$\ell_1$@300",
    r"$\ell_2$@2",
    r"$\ell_2$@8",
    r"$\ell_\infty$@8",
    r"$\ell_\infty$@30",
    "sev@3",
]

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

TARGETS = {
    "google_nothink": "Gemini Flash (no think)",
    "openai":         "GPT-4o",
}

# ── Surrogate family grouping → marker shape ──────────────────────────────────
def surrogate_family(s):
    if s.startswith("CLIP"):        return "CLIP"
    if s.startswith("MetaCLIP"):    return "MetaCLIP"
    if s.startswith("SigLIP"):      return "SigLIP2"
    return "Other"

FAMILY_MARKERS = {
    "CLIP":     "o",       # circle
    "MetaCLIP": "D",       # diamond
    "SigLIP2":  "*",       # star
    "Other":    "s",       # square
}

# One distinct color per surrogate model
SURROGATE_COLORS = {
    "CLIP B/16":       "#4E79A7",
    "CLIP H/14":       "#A0CBE8",
    "MetaCLIP H/14":   "#59A14F",
    "SigLIP2 base":    "#E05759",
    "SigLIP2 SO400M":  "#FF9D9A",
    "SigLIP2 NaFlex":  "#F1CE63",
    "Common corr.":    "#9C755F",
}

TARGET_COLORS = {
    "google_nothink": "#3DBD8A",   # teal-green  (Gemini)
    "openai":         "#5B9BD5",   # steel-blue  (GPT-4o)
}

BASE = Path("llm_classification_results")

# ── Helpers ───────────────────────────────────────────────────────────────────
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
            if rec.get("error"):
                continue
            total += 1
            if rec.get("correct", False):
                correct += 1
        except Exception:
            pass
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
    if not recs:
        return None
    seen = {}
    for r in recs:
        if not r.get("error") and r["index"] not in seen:
            seen[r["index"]] = r
    total   = len(seen)
    correct = sum(1 for r in seen.values() if r.get("correct", False))
    return (correct, total) if total > 0 else None

# ── Load all results ──────────────────────────────────────────────────────────
# acc[(surrogate, pert_cond, target_key, dataset)] = float
acc: dict[tuple, float] = {}

for mp in sorted(BASE.glob("batch_manifest__all_datasets__*.json")):
    exp = mp.stem.replace("batch_manifest__all_datasets__", "")
    if exp not in EXPERIMENT_META:
        continue
    surrogate, pert = EXPERIMENT_META[exp]
    manifest = json.loads(mp.read_text())

    groups: dict[tuple, list] = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)

    for (ds, key), entries in groups.items():
        if key not in TARGETS:
            continue
        main_entries = [e for e in entries if "__complement" not in e["run_name"]]
        comp_entries = [e for e in entries if "__complement"     in e["run_name"]]
        main_ok      = [e for e in main_entries if e["status"] != "failed"]
        main         = main_ok[0] if main_ok else (main_entries[0] if main_entries else entries[0])
        comp         = comp_entries[0] if comp_entries else None
        if main["status"] != "retrieved":
            continue
        result = (
            merge_complement(main["run_name"], comp["run_name"])
            if comp and comp["status"] == "retrieved"
            else load_predictions(main["run_name"])
        )
        if result:
            correct, total = result
            acc[(surrogate, pert, key, ds)] = correct / total

# ── Build PCA feature matrix ──────────────────────────────────────────────────
# One row per (surrogate, target_key).
# Feature vector: mean-over-datasets accuracy for each perturbation condition.
surrogates   = sorted({s for (s, p, t, d) in acc})
target_keys  = sorted({t for (s, p, t, d) in acc})

point_labels = [(s, t) for s in surrogates for t in target_keys]

X = np.full((len(point_labels), len(PERT_ORDER)), np.nan)
for row_i, (s, t) in enumerate(point_labels):
    for col_j, pert in enumerate(PERT_ORDER):
        vals = [acc.get((s, pert, t, ds), np.nan) for ds in DATASETS]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            X[row_i, col_j] = np.mean(vals)

# Drop rows with all-NaN (surrogate × target combos with zero data)
valid_mask = ~np.all(np.isnan(X), axis=1)
X_valid    = X[valid_mask]
labels_valid = [point_labels[i] for i in range(len(point_labels)) if valid_mask[i]]

if len(labels_valid) < 2:
    print("Not enough data for PCA yet — run more experiments first.")
    exit()

# Column-mean imputation for remaining NaNs
col_means = np.nanmean(X_valid, axis=0)
for i in range(X_valid.shape[0]):
    for j in range(X_valid.shape[1]):
        if np.isnan(X_valid[i, j]):
            X_valid[i, j] = col_means[j]

pca    = PCA(n_components=2)
coords = pca.fit_transform(X_valid)
var    = pca.explained_variance_ratio_

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

fig, ax = plt.subplots(figsize=(3.5, 3.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F7F7F7")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", length=2)
ax.grid(True, color="white", linewidth=0.8, zorder=0)

seen_surrogates = set()
seen_targets    = set()
seen_families   = set()

for i, (s, t) in enumerate(labels_valid):
    xy      = coords[i]
    family  = surrogate_family(s)
    color   = TARGET_COLORS[t]
    marker  = FAMILY_MARKERS[family]

    # Size encodes mean accuracy across all conditions for this (surrogate, target)
    row_vals = X_valid[i]
    mean_acc = np.mean(row_vals[~np.isnan(row_vals)]) if not np.all(np.isnan(row_vals)) else 0.5
    size = 20 + 80 * mean_acc   # 20–100 pt²

    ax.scatter(
        xy[0], xy[1],
        s=size, c=color, marker=marker,
        edgecolors="white", linewidths=0.5,
        alpha=0.90, zorder=4,
    )

    # Label each point with short surrogate name
    short = (s.replace("SigLIP2 ", "SiG2-")
              .replace("MetaCLIP ", "MC-")
              .replace("CLIP ", "CL-")
              .replace("Common corr.", "Corr."))
    ax.annotate(
        short, xy,
        xytext=(4, 3), textcoords="offset points",
        fontsize=NEURIPS_FONTSIZE - 2.5, color="#333333",
        zorder=5,
    )

    seen_surrogates.add(s)
    seen_targets.add(t)
    seen_families.add(family)

# Draw convex-hull ellipses per target to show clustering
from matplotlib.patches import Ellipse
for t, color in TARGET_COLORS.items():
    pts = np.array([coords[i] for i, (s, tk) in enumerate(labels_valid) if tk == t])
    if len(pts) < 3:
        continue
    mu  = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals_eig, vecs = np.linalg.eigh(cov)
    order = vals_eig.argsort()[::-1]
    vals_eig, vecs = vals_eig[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * 1.8 * np.sqrt(vals_eig)   # ~1.8σ ellipse
    ell = Ellipse(mu, w, h, angle=angle,
                  color=color, alpha=0.10, zorder=2, linewidth=0)
    ax.add_patch(ell)
    ell2 = Ellipse(mu, w, h, angle=angle,
                   fill=False, edgecolor=color, alpha=0.35,
                   linewidth=0.8, linestyle="--", zorder=3)
    ax.add_patch(ell2)

ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% variance)", labelpad=3)
ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% variance)", labelpad=3)
ax.set_title("Robustness profile PCA\n(perturbation conditions as features)",
             fontsize=NEURIPS_FONTSIZE, pad=4)

# ── Legends ───────────────────────────────────────────────────────────────────
# Legend 1: color = target model
target_handles = [
    mpatches.Patch(color=TARGET_COLORS[k], label=TARGETS[k], alpha=0.85)
    for k in TARGET_COLORS if k in seen_targets
]
leg1 = ax.legend(
    handles=target_handles,
    title="Target model",
    title_fontsize=NEURIPS_FONTSIZE - 1,
    loc="upper left",
    frameon=True, framealpha=0.9, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.45, labelspacing=0.25,
)
ax.add_artist(leg1)

# Legend 2: shape = surrogate family
family_handles = [
    mlines.Line2D([], [], color="#555555",
                  marker=FAMILY_MARKERS[f], linestyle="None",
                  markersize=5, label=f)
    for f in FAMILY_MARKERS if f in seen_families
]
# Size legend
size_handles = [
    mlines.Line2D([], [], color="#888888",
                  marker="o", linestyle="None",
                  markersize=np.sqrt(20 + 80 * v),
                  label=f"{int(v*100)}%")
    for v in [0.3, 0.6, 0.9]
]
leg2 = ax.legend(
    handles=family_handles + [mlines.Line2D([], [], linestyle="None")] + size_handles,
    title="Family / mean acc.",
    title_fontsize=NEURIPS_FONTSIZE - 1,
    loc="lower right",
    frameon=True, framealpha=0.9, edgecolor="none",
    handlelength=0.9, handletextpad=0.4,
    borderpad=0.45, labelspacing=0.25,
)

plt.tight_layout(pad=0.5)

out = Path("./results_analysis_neurips2026/transfer_pca.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()