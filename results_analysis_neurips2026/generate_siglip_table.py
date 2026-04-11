#!/usr/bin/env python3
"""
SigLIP2 sub-study: linf@30, 3 datasets only.
Same presentation logic as the main table:
  - \toprule/\midrule/\bottomrule
  - 3-level header: target super-group / Clean+Adv sub-group / surrogate labels
  - rotated surrogate column headers
  - bold avg row via \textbf{}, bold avg cells
  - Dataset || [Gemini: Clean | base | SO400M | NaFlex] || [GPT-4o: Clean | base | SO400M | NaFlex]
"""

import json
import numpy as np
from pathlib import Path

EXPERIMENT_META = {
    "test_v1":                    ("__clean__",      "__clean__"),
    "adv_siglip2_linf30":         ("SigLIP2 base",   "linf@30"),
    "adv_siglip2_384_linf_eps30": ("SigLIP2 SO400M", "linf@30"),
    "adv_siglip2_correct_naflex_linf_eps30": ("SigLIP2 NaFlex", "linf@30"),
}

SURROGATE_ORDER = ["SigLIP2 base", "SigLIP2 SO400M", "SigLIP2 NaFlex"]
SURR_SHORT = {
    "SigLIP2 base":   "base",
    "SigLIP2 SO400M": "SO400M",
    "SigLIP2 NaFlex": "NaFlex",
}

DATASETS = [
    "caltech101",
    "flowers-102",
    "uc-merced-land-use-dataset",
]
DS_LABEL = {
    "caltech101":                 "Caltech-101",
    "flowers-102":                "Flowers-102",
    "uc-merced-land-use-dataset": "UC Merced",
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

def get(surr, tgt, ds):
    return raw.get((surr, "linf@30", tgt, ds), np.nan)

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
# Dataset || [Clean | base | SO400M | NaFlex] || [Clean | base | SO400M | NaFlex]
# same separator logic as main table: || between target blocks, | between clean and adv
N_SURR = len(SURROGATE_ORDER)   # 3
# l || r | rrr || r | rrr
COL_SPEC = r"l || r | rrr || r | rrr"

lines = []
A = lambda s: lines.append(s)

A(r"% Requires: booktabs, multirow")
A(r"\begin{table}[t]")
A(r"\centering")
A(r"\setlength{\tabcolsep}{5pt}")
A(r"\renewcommand{\arraystretch}{1.08}")
A(r"\small")
A(r"\caption{%")
A(r"  Transfer accuracy (\%) of SigLIP2 variants under $\ell_\infty^{30}$")
A(r"  for each target model, on coarse-grained datasets.")
A(r"  Clean accuracy reported alongside each target block for reference.")
A(r"  {\textemdash}: not evaluated.")
A(r"}")
A(r"\label{tab:siglip2_linf30}")
A(r"\begin{tabular}{" + COL_SPEC + r"}")
A(r"\toprule")

# ── Header row 1: target super-groups ────────────────────────────────────────
# Each block spans 1 (clean) + N_SURR (adv) = 4 columns
N_BLOCK = 1 + N_SURR
A(r"\multirow{2}{*}{\textbf{Dataset}}"
  r" & \multicolumn{" + str(N_BLOCK) + r"}{c||}"
  r"{\textbf{" + TARGET_LABELS["google_nothink"] + r"}}"
  r" & \multicolumn{" + str(N_BLOCK) + r"}{c}"
  r"{\textbf{" + TARGET_LABELS["openai"] + r"}}"
  r" \\")

# ── Header row 2: Clean | surrogate names (rotated, matching main table style)
surr_heads = " & ".join(
    r"\textit{" + SURR_SHORT[s] + r"}"
    for s in SURROGATE_ORDER
)
A(r" & Clean & " + surr_heads +
  r" & Clean & " + surr_heads + r" \\")
A(r"\midrule")

# ── Data rows ─────────────────────────────────────────────────────────────────
for ds in DATASETS:
    cells = [DS_LABEL[ds]]
    for tgt in TARGET_ORDER:
        cells.append(fmt(get_clean(tgt, ds)))
        for surr in SURROGATE_ORDER:
            cells.append(fmt(get(surr, tgt, ds)))
    A(" & ".join(cells) + r" \\")

# ── Average row ───────────────────────────────────────────────────────────────
A(r"\midrule")
avg_cells = [r"\textbf{Average}"]
for tgt in TARGET_ORDER:
    avg_cells.append(fmt(nanmean([get_clean(tgt, ds) for ds in DATASETS]), bold=True))
    for surr in SURROGATE_ORDER:
        avg_cells.append(fmt(nanmean([get(surr, tgt, ds) for ds in DATASETS]), bold=True))
A(" & ".join(avg_cells) + r" \\")

A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")

output = "\n".join(lines)
out = Path("./results_analysis_neurips2026/siglip2_linf30_table.tex")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(output)
print(f"Saved → {out}")
print()
print(output)