#!/usr/bin/env python3
"""
plot_transfer_robustness.py

Visualises adversarial transferability across:
  - Surrogate VLMs  (rows)
  - Perturbation norms / budgets  (columns)
  - Target models  (sub-panels: Gemini Flash no-think vs GPT-4o)

Layout: one heatmap per target model, side by side.
Cells show mean accuracy over datasets (lower = stronger transfer).
Grey cells = pending / missing data.

Save path: ./results_analysis_neurips2026/transfer_robustness.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Experiment registry ───────────────────────────────────────────────────────
# Maps manifest stem → (surrogate_label, norm_label)
# Add / remove entries as your runs complete.
EXPERIMENT_META = {
    # CLIP ViT-B/16
    "adv_clip_l1_eps75":          ("CLIP B/16",      r"$\ell_1$-75"),
    "adv_clip_l1_eps300":         ("CLIP B/16",      r"$\ell_1$-300"),
    "adv_clip_l2_eps2":           ("CLIP B/16",      r"$\ell_2$-2"),
    "adv_clip_l2_eps8":           ("CLIP B/16",      r"$\ell_2$-8"),
    "adv_clip_linf8":             ("CLIP B/16",      r"$\ell_\infty$-8"),
    "adv_linf30":                 ("CLIP B/16",      r"$\ell_\infty$-30"),
    # CLIP ViT-H/14
    "adv_clip_vith14_l1_eps75":   ("CLIP H/14",      r"$\ell_1$-75"),
    "adv_clip_vith14_l1_eps300":  ("CLIP H/14",      r"$\ell_1$-300"),
    "adv_clip_vith14_l2_eps2":    ("CLIP H/14",      r"$\ell_2$-2"),
    "adv_clip_vith14_l2_eps8":    ("CLIP H/14",      r"$\ell_2$-8"),
    "adv_clip_vith14_linf8":      ("CLIP H/14",      r"$\ell_\infty$-8"),
    "adv_clip_vith14_linf30":     ("CLIP H/14",      r"$\ell_\infty$-30"),
    # MetaCLIP ViT-H/14
    "adv_metaclip_l1_eps75":      ("MetaCLIP H/14",  r"$\ell_1$-75"),
    "adv_metaclip_l1_eps300":     ("MetaCLIP H/14",  r"$\ell_1$-300"),
    "adv_metaclip_l2_eps2":       ("MetaCLIP H/14",  r"$\ell_2$-2"),
    "adv_metaclip_l2_eps8":       ("MetaCLIP H/14",  r"$\ell_2$-8"),
    "adv_metaclip_linf_eps8":     ("MetaCLIP H/14",  r"$\ell_\infty$-8"),
    "adv_metaclip_linf_eps30":    ("MetaCLIP H/14",  r"$\ell_\infty$-30"),
    # SigLIP2 base
    "adv_siglip2_linf8":          ("SigLIP2 base",   r"$\ell_\infty$-8"),
    "adv_siglip2_linf30":         ("SigLIP2 base",   r"$\ell_\infty$-30"),
    # SigLIP2 SO400M-384
    "adv_siglip2_384_l1_eps75":   ("SigLIP2 SO400M", r"$\ell_1$-75"),
    "adv_siglip2_384_l1_eps300":  ("SigLIP2 SO400M", r"$\ell_1$-300"),
    "adv_siglip2_384_l2_eps2":    ("SigLIP2 SO400M", r"$\ell_2$-2"),
    "adv_siglip2_384_l2_eps8":    ("SigLIP2 SO400M", r"$\ell_2$-8"),
    "adv_siglip2_384_linf_eps8":  ("SigLIP2 SO400M", r"$\ell_\infty$-8"),
    "adv_siglip2_384_linf_eps30": ("SigLIP2 SO400M", r"$\ell_\infty$-30"),
    # SigLIP2 NaFlex
    "adv_siglip2_correct_naflex_linf_eps30": ("SigLIP2 NaFlex", r"$\ell_\infty$-30"),
    # Common corruptions (not adversarial — shown separately at end)
    "common_severity3":           ("Common corr.",   "severity-3"),
}

# Canonical display order
SURROGATE_ORDER = [
    "CLIP B/16",
    "CLIP H/14",
    "MetaCLIP H/14",
    "SigLIP2 base",
    "SigLIP2 SO400M",
    "SigLIP2 NaFlex",
    "Common corr.",
]

NORM_ORDER = [
    r"$\ell_1$-75",
    r"$\ell_1$-300",
    r"$\ell_2$-2",
    r"$\ell_2$-8",
    r"$\ell_\infty$-8",
    r"$\ell_\infty$-30",
    "severity-3",
]

# Target model key mapping  (key in predictions → display label)
TARGETS = {
    "google_nothink": "Gemini Flash\n(no think)",
    "openai":         "GPT-4o",
}

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

BASE = Path("llm_classification_results")

# ── Helpers (same logic as summarize_results.py) ──────────────────────────────

def dataset_key(run_name: str) -> str:
    for ds in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(ds):
            return ds
    return run_name.split("__")[0]


def load_predictions(run_name: str) -> tuple[int, int] | None:
    pred_path = BASE / run_name / "predictions.jsonl"
    if not pred_path.exists():
        return None
    correct = total = 0
    for line in pred_path.read_text().splitlines():
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


def merge_complement(main_run: str, comp_run: str) -> tuple[int, int] | None:
    recs = []
    for run in [main_run, comp_run]:
        p = BASE / run / "predictions.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    if not recs:
        return None
    seen = {}
    for r in recs:
        if not r.get("error") and r["index"] not in seen:
            seen[r["index"]] = r
    total   = len(seen)
    correct = sum(1 for r in seen.values() if r.get("correct", False))
    return (correct, total) if total > 0 else None


# ── Load all manifests ─────────────────────────────────────────────────────────
# results[(surrogate, norm, target_key)] = list of acc floats (one per dataset)
results: dict[tuple, list[float]] = {}

manifest_files = sorted(BASE.glob("batch_manifest__all_datasets__*.json"))

for mp in manifest_files:
    exp = mp.stem.replace("batch_manifest__all_datasets__", "")
    if exp not in EXPERIMENT_META:
        continue
    surrogate, norm = EXPERIMENT_META[exp]

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
            acc = correct / total
            results.setdefault((surrogate, norm, key), []).append(acc)

# Average over datasets
mean_acc: dict[tuple, float] = {
    k: float(np.mean(v)) for k, v in results.items()
}

# ── Build grid matrices ───────────────────────────────────────────────────────
# Filter to surrogates / norms that actually appear in the data
present_surrogates = [s for s in SURROGATE_ORDER
                      if any(k[0] == s for k in mean_acc)]
present_norms      = [n for n in NORM_ORDER
                      if any(k[1] == n for k in mean_acc)]

nS, nN = len(present_surrogates), len(present_norms)

grids   = {}   # target_key → (nS × nN) float array, nan = missing
for tgt in TARGETS:
    g = np.full((nS, nN), np.nan)
    for i, s in enumerate(present_surrogates):
        for j, n in enumerate(present_norms):
            v = mean_acc.get((s, n, tgt), np.nan)
            g[i, j] = v
    grids[tgt] = g

# ── NeurIPS-style plotting ────────────────────────────────────────────────────
NEURIPS_FONTSIZE = 7
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      NEURIPS_FONTSIZE,
    "axes.labelsize": NEURIPS_FONTSIZE,
    "xtick.labelsize":NEURIPS_FONTSIZE - 1,
    "ytick.labelsize":NEURIPS_FONTSIZE - 1,
    "figure.dpi":     300,
})

n_targets = len(TARGETS)
# Full-width NeurIPS figure (two heatmaps side by side)
fig, axes = plt.subplots(
    1, n_targets,
    figsize=(6.75, max(2.2, 0.32 * nS + 0.8)),
    sharey=True,
)
fig.patch.set_facecolor("white")

# Colormap: white (100 %) → deep red (0 %)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rob", ["#C0392B", "#E8A090", "#FDECEA", "#FFFFFF"], N=256
)
cmap.set_bad(color="#CCCCCC")   # grey for NaN

vmin, vmax = 0.0, 1.0

for ax, (tgt_key, tgt_label) in zip(axes, TARGETS.items()):
    g = grids[tgt_key]

    # Mask full-NaN rows (surrogate never evaluated against this target)
    display = np.ma.masked_invalid(g)

    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(nS):
        for j in range(nN):
            val = g[i, j]
            if not np.isnan(val):
                txt   = f"{val*100:.1f}"
                color = "white" if val < 0.45 else "#222222"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=NEURIPS_FONTSIZE - 2, color=color)
            else:
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=NEURIPS_FONTSIZE - 2, color="#888888")

    ax.set_xticks(range(nN))
    ax.set_xticklabels(present_norms, rotation=40, ha="right")
    ax.set_title(tgt_label, fontsize=NEURIPS_FONTSIZE, pad=4)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    # Subtle grid lines between cells
    for x in np.arange(-0.5, nN, 1):
        ax.axvline(x, color="white", linewidth=0.6, zorder=3)
    for y in np.arange(-0.5, nS, 1):
        ax.axhline(y, color="white", linewidth=0.6, zorder=3)

axes[0].set_yticks(range(nS))
axes[0].set_yticklabels(present_surrogates)
axes[0].set_ylabel("Surrogate", labelpad=4)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                    fraction=0.03, pad=0.02, shrink=0.85)
cbar.set_label("Avg. accuracy", fontsize=NEURIPS_FONTSIZE - 1, labelpad=3)
cbar.ax.tick_params(labelsize=NEURIPS_FONTSIZE - 2, length=2)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

fig.suptitle("Adversarial transfer: accuracy under attack (↓ = stronger transfer)",
             fontsize=NEURIPS_FONTSIZE, y=1.01)

plt.tight_layout(pad=0.5)

out = Path("./results_analysis_neurips2026/transfer_robustness.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()