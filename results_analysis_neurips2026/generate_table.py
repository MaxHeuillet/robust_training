#!/usr/bin/env python3
"""
LaTeX table — unified layout:
  rows    = surrogate blocks stacked vertically,
            each block: one row per dataset + italic avg row
  columns = Dataset | [Gemini: Clean | adv... | Sev3 | Avg] | [GPT-4o: same]
  one table*, requires booktabs + multirow + array
"""

import json
import numpy as np
from pathlib import Path

# ── Data ─────────────────────────────────────────────────────────────────────
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

DATASETS = [
    "caltech101", "fgvc-aircraft-2013b", "flowers-102",
    "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset",
]
DS_LABEL = {
    "caltech101":                 "Caltech-101",
    "fgvc-aircraft-2013b":        "FGVC Aircraft",
    "flowers-102":                "Flowers-102",
    "oxford-iiit-pet":            "Oxford Pet",
    "stanford_cars":              "Stanford Cars",
    "uc-merced-land-use-dataset": "UC Merced",
}

SURROGATE_ORDER = [
    "CLIP B/16", "CLIP H/14", "MetaCLIP H/14", "SigLIP2 SO400M", 
] # "SigLIP2 base",  "SigLIP2 NaFlex",
SURR_LABEL = {
    "CLIP B/16":       r"CLIP ViT-B/16",
    "CLIP H/14":       r"CLIP ViT-H/14",
    "MetaCLIP H/14":   r"MetaCLIP ViT-H/14",
    "SigLIP2 base":    r"SigLIP2 base",
    "SigLIP2 SO400M":  r"SigLIP2 SO400M",
    "SigLIP2 NaFlex":  r"SigLIP2 NaFlex",
}

# Canonical perturbation order (sev@3 handled separately)
ADV_PERTS = ["l1@75", "l1@300", "l2@2", "l2@8", "linf@8", "linf@30"]
ALL_PERTS = ADV_PERTS + ["sev@3"]
PERT_LABEL = {
    "l1@75":   r"$\ell_1^{75}$",
    "l1@300":  r"$\ell_1^{300}$",
    "l2@2":    r"$\ell_2^{2}$",
    "l2@8":    r"$\ell_2^{8}$",
    "linf@8":  r"$\ell_\infty^{8}$",
    "linf@30": r"$\ell_\infty^{30}$",
    "sev@3":   r"Sev.$^3$",
}

TARGET_ORDER  = ["google_nothink", "openai"]
TARGET_LABELS = {
    "google_nothink": "Gemini Flash (no think)",
    "openai":         "GPT-4o",
}

BASE = Path("llm_classification_results")

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

raw = {}
for mp in sorted(BASE.glob("batch_manifest__all_datasets__*.json")):
    exp = mp.stem.replace("batch_manifest__all_datasets__","")
    if exp not in EXPERIMENT_META: continue
    surr, pert = EXPERIMENT_META[exp]
    manifest = json.loads(mp.read_text())
    groups = {}
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
        if res: raw[(surr, pert, key, ds)] = res[0]/res[1]

def get(surr, pert, tgt, ds):
    src = "__common__" if pert == "sev@3" else surr
    return raw.get((src, pert, tgt, ds), np.nan)

def get_clean(tgt, ds):
    return raw.get(("__clean__", "__clean__", tgt, ds), np.nan)

def nanmean(lst):
    v = [x for x in lst if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan

MISS = r"{\textemdash}"

def fmt(v, bold=False):
    if np.isnan(v): return MISS
    s = f"{v*100:.1f}"
    return r"\textbf{" + s + r"}" if bold else s

# ── Column spec ───────────────────────────────────────────────────────────────
# Dataset  ||  [Clean | adv×6 | sev3 | Avg]  ||  [Clean | adv×6 | sev3 | Avg]
# per-target block: 1 + 6 + 1 + 1 = 9 cols
# total: 1 (dataset) + 2 × 9 = 19 cols
# col spec:  l  ||  r | rrrrrr r | r  ||  r | rrrrrr r | r
N_PER_TGT   = 1 + len(ADV_PERTS) + 1 + 1   # clean + adv + sev + avg = 9
N_TOTAL     = 1 + len(TARGET_ORDER) * N_PER_TGT

tgt_col_spec = r"r | rrrrrr r | r"   # clean | 6 adv | sev | avg
COL_SPEC = r"l || " + r" || ".join([tgt_col_spec] * len(TARGET_ORDER))

# ── Build table ───────────────────────────────────────────────────────────────
lines = []
A = lambda s: lines.append(s)

A(r"% Requires: booktabs, multirow, array")
A(r"\begin{table*}[t]")
A(r"\centering")
A(r"\setlength{\tabcolsep}{3.5pt}")
A(r"\renewcommand{\arraystretch}{1.08}")
A(r"\small")
A(r"\caption{%")
A(r"  Adversarial transfer accuracy (\%) under each surrogate and threat model,")
A(r"  evaluated on two closed-source VLMs (column super-groups).")
A(r"  \emph{Clean}: no attack.")
A(r"  Sev.$^3$: common corruptions at severity 3 (surrogate-independent).")
A(r"  Bold values = column/block averages.")
A(r"  {\textemdash}: not evaluated.")
A(r"}")
A(r"\label{tab:transfer_full}")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{" + COL_SPEC + r"}")
A(r"\toprule")

# ── Header row 1: target super-group labels ───────────────────────────────────
h1_cells = [r"\multirow{3}{*}{\textbf{Dataset}}"]
for tgt in TARGET_ORDER:
    h1_cells.append(
        r"& \multicolumn{" + str(N_PER_TGT) + r"}{c||}{\textbf{"
        + TARGET_LABELS[tgt] + r"}}"
    )
# fix last target: no trailing ||
h1_cells[-1] = h1_cells[-1].replace(r"c||", r"c")
A("  ".join(h1_cells) + r" \\")

# ── Header row 2: Clean | Adversarial | Sev | Avg sub-labels ─────────────────
h2_cells = [""]   # empty for dataset multirow
for tgt in TARGET_ORDER:
    h2_cells.append(
        r"& \multirow{2}{*}{Clean}"
        r" & \multicolumn{6}{c|}{Adversarial}"
        r" & \multirow{2}{*}{Sev.$^3$}"
        r" & \multirow{2}{*}{\textbf{Avg.}}"
    )
A("  ".join(h2_cells) + r" \\")

# ── Header row 3: individual perturbation labels ──────────────────────────────
h3_cells = [""]
adv_heads = " & ".join(PERT_LABEL[p] for p in ADV_PERTS)
for tgt in TARGET_ORDER:
    h3_cells.append(r"& & " + adv_heads + r" & &")
A("  ".join(h3_cells) + r" \\")
A(r"\midrule")

# ── Surrogate blocks ──────────────────────────────────────────────────────────
for bi, surr in enumerate(SURROGATE_ORDER):

    # Block header spanning all columns
    A(r"\multicolumn{" + str(N_TOTAL) + r"}{l}{"
      r"\textit{Surrogate: \textbf{" + SURR_LABEL[surr] + r"}}}  \\")
    A(r"\addlinespace[1pt]")

    # Dataset rows
    surr_adv_vals = {tgt: [] for tgt in TARGET_ORDER}
    for ds in DATASETS:
        row_cells = [r"\hspace{4pt}" + DS_LABEL[ds]]
        for tgt in TARGET_ORDER:
            c_val = get_clean(tgt, ds)
            row_cells.append(fmt(c_val))
            ds_adv = []
            for p in ADV_PERTS:
                v = get(surr, p, tgt, ds)
                row_cells.append(fmt(v))
                if not np.isnan(v):
                    ds_adv.append(v)
                    surr_adv_vals[tgt].append(v)
            # sev@3
            sev = get(surr, "sev@3", tgt, ds)
            row_cells.append(fmt(sev))
            if not np.isnan(sev):
                surr_adv_vals[tgt].append(sev)
            # row avg (adv + sev, not clean)
            row_adv = ds_adv + ([sev] if not np.isnan(sev) else [])
            row_cells.append(fmt(nanmean(row_adv)))
        A(" & ".join(row_cells) + r" \\")

    # Surrogate average row
    A(r"\addlinespace[1pt]")
    avg_cells = [r"\hspace{4pt}\textit{Avg.}"]
    for tgt in TARGET_ORDER:
        # clean avg
        clean_vals = [get_clean(tgt, ds) for ds in DATASETS]
        avg_cells.append(fmt(nanmean(clean_vals), bold=True))
        # per-pert avg
        for p in ADV_PERTS:
            col = [get(surr, p, tgt, ds) for ds in DATASETS]
            avg_cells.append(fmt(nanmean(col), bold=True))
        # sev@3 avg
        sev_col = [get(surr, "sev@3", tgt, ds) for ds in DATASETS]
        avg_cells.append(fmt(nanmean(sev_col), bold=True))
        # block avg (all adv incl sev)
        avg_cells.append(fmt(nanmean(surr_adv_vals[tgt]), bold=True))
    A(" & ".join(avg_cells) + r" \\")

    if bi < len(SURROGATE_ORDER) - 1:
        A(r"\midrule")

# ── Global average row ────────────────────────────────────────────────────────
A(r"\midrule")
glob_cells = [r"\textbf{Global avg.}"]
for tgt in TARGET_ORDER:
    clean_all = [get_clean(tgt, ds) for ds in DATASETS]
    glob_cells.append(fmt(nanmean(clean_all), bold=True))
    all_adv = []
    for p in ADV_PERTS:
        col = [get(surr, p, tgt, ds)
               for surr in SURROGATE_ORDER for ds in DATASETS]
        glob_cells.append(fmt(nanmean(col), bold=True))
        all_adv.extend(col)
    sev_col = [get("__any__", "sev@3", tgt, ds) for ds in DATASETS]
    glob_cells.append(fmt(nanmean(sev_col), bold=True))
    all_adv.extend(sev_col)
    glob_cells.append(fmt(nanmean(all_adv), bold=True))
A(" & ".join(glob_cells) + r" \\")

A(r"\bottomrule")
A(r"\end{tabular}")
A(r"}")   # resizebox
A(r"\end{table*}")

output = "\n".join(lines)

out = Path("./results_analysis_neurips2026/transfer_table.tex")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(output)
print(f"Saved → {out}")