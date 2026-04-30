#!/usr/bin/env python3
"""
generate_table_final.py

Two outputs:
  1. transfer_table.tex / .csv  — adversarial only (no sev@3 column)
  2. common_table.tex           — standalone table for common corruptions
"""

import json, csv
import numpy as np
from pathlib import Path

# ── Experiment registry ───────────────────────────────────────────────────────
EXPERIMENT_META = {
    "test_v1":                    ("__clean__",      "__clean__"),
    "adv_clip_l1_eps75":          ("CLIP B/16",      "l1@75"),
    "adv_clip_l1_eps300":         ("CLIP B/16",      "l1@300"),
    "adv_clip_l2_eps2":           ("CLIP B/16",      "l2@2"),
    "adv_clip_l2_eps8":           ("CLIP B/16",      "l2@8"),
    "adv_clip_linf_eps4":             ("CLIP B/16",      "linf@4"),
    "adv_clip_linf8":             ("CLIP B/16",      "linf@8"),
    "adv_linf30":                 ("CLIP B/16",      "linf@30"),
    "adv_clip_vith14_l1_eps75":   ("CLIP H/14",      "l1@75"),
    "adv_clip_vith14_l1_eps300":  ("CLIP H/14",      "l1@300"),
    "adv_clip_vith14_l2_eps2":    ("CLIP H/14",      "l2@2"),
    "adv_clip_vith14_l2_eps8":    ("CLIP H/14",      "l2@8"),
    "adv_clip_vith14_linf_eps4":      ("CLIP H/14",      "linf@4"),
    "adv_clip_vith14_linf8":      ("CLIP H/14",      "linf@8"),
    "adv_clip_vith14_linf30":     ("CLIP H/14",      "linf@30"),
    "adv_metaclip_l1_eps75":      ("MetaCLIP H/14",  "l1@75"),
    "adv_metaclip_l1_eps300":     ("MetaCLIP H/14",  "l1@300"),
    "adv_metaclip_l2_eps2":       ("MetaCLIP H/14",  "l2@2"),
    "adv_metaclip_l2_eps8":       ("MetaCLIP H/14",  "l2@8"),
    "adv_metaclip_linf_eps4":     ("MetaCLIP H/14",  "linf@4"),
    "adv_metaclip_linf_eps8":     ("MetaCLIP H/14",  "linf@8"),
    "adv_metaclip_linf_eps30":    ("MetaCLIP H/14",  "linf@30"),
    "adv_siglip2_linf4":          ("SigLIP2 base",   "linf@4"),
    "adv_siglip2_linf8":          ("SigLIP2 base",   "linf@8"),
    "adv_siglip2_linf30":         ("SigLIP2 base",   "linf@30"),
    "adv_siglip2_384_l1_eps75":   ("SigLIP2 SO400M", "l1@75"),
    "adv_siglip2_384_l1_eps300":  ("SigLIP2 SO400M", "l1@300"),
    "adv_siglip2_384_l2_eps2":    ("SigLIP2 SO400M", "l2@2"),
    "adv_siglip2_384_l2_eps8":    ("SigLIP2 SO400M", "l2@8"),
    "adv_siglip2_384_linf_eps4":  ("SigLIP2 SO400M", "linf@4"),
    "adv_siglip2_384_linf_eps8":  ("SigLIP2 SO400M", "linf@8"),
    "adv_siglip2_384_linf_eps30": ("SigLIP2 SO400M", "linf@30"),
    "adv_siglip2_correct_naflex_linf_eps4":  ("SigLIP2 NaFlex", "linf@4"),
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
]
SURR_LABEL = {
    "CLIP B/16":       r"CLIP ViT-B/16",
    "CLIP H/14":       r"CLIP ViT-H/14",
    "MetaCLIP H/14":   r"MetaCLIP ViT-H/14",
    "SigLIP2 base":    r"SigLIP2 base",
    "SigLIP2 SO400M":  r"SigLIP2 SO400M",
    "SigLIP2 NaFlex":  r"SigLIP2 NaFlex",
}

# Adversarial only — sev@3 removed from main table
ADV_PERTS = ["l1@75", "l1@300", "l2@2", "l2@8", "linf@4", "linf@8", "linf@30", "sev@3"]

PERT_LABEL = {
    "l1@75":   r"$\ell_1^{75}$",
    "l1@300":  r"$\ell_1^{300}$",
    "l2@2":    r"$\ell_2^{2}$",
    "l2@8":    r"$\ell_2^{8}$",
    "linf@4":  r"$\ell_\infty^{4}$",
    "linf@8":  r"$\ell_\infty^{8}$",
    "linf@30": r"$\ell_\infty^{30}$",
    "sev@3":   r"$\text{sev@3}$",
}

TARGET_ORDER  = ["google_nothink", "openai"]
TARGET_LABELS = {
    "google_nothink": "Gemini Flash (no think)",
    "openai":         "GPT-4o",
}

BASE = Path("llm_classification_results")

# ── Loaders ───────────────────────────────────────────────────────────────────
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
    exp = mp.stem.replace("batch_manifest__all_datasets__", "")
    if exp not in EXPERIMENT_META: continue
    surr, pert = EXPERIMENT_META[exp]
    manifest = json.loads(mp.read_text())
    groups = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)
    for (ds, key), entries in groups.items():
        if key not in {"google_nothink", "openai"}: continue
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

OUT = Path("./results_analysis_neurips2026")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Adversarial transfer (no sev@3)
# ══════════════════════════════════════════════════════════════════════════════
N_ADV     = len(ADV_PERTS)           # 7
N_PER_TGT = 1 + N_ADV + 1           # clean + 7 adv + avg  (no sev col)
N_TOTAL   = 1 + len(TARGET_ORDER) * N_PER_TGT

tgt_col_spec = r"r | " + "r" * N_ADV + r" | r"
COL_SPEC = r"l || " + r" || ".join([tgt_col_spec] * len(TARGET_ORDER))

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
A(r"  Bold values = column/block averages.")
A(r"  {\textemdash}: not evaluated.")
A(r"  See \cref{tab:common} for common corruption results.")
A(r"}")
A(r"\label{tab:transfer_full}")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{" + COL_SPEC + r"}")
A(r"\toprule")

# Header row 1: target super-groups
h1 = [r"\multirow{3}{*}{\textbf{Dataset}}"]
for tgt in TARGET_ORDER:
    cell = (r"& \multicolumn{" + str(N_PER_TGT) + r"}{c||}{\textbf{"
            + TARGET_LABELS[tgt] + r"}}")
    h1.append(cell)
h1[-1] = h1[-1].replace(r"c||", r"c")
A("  ".join(h1) + r" \\")

# Header row 2: Clean | Adversarial | Avg
h2 = [""]
for tgt in TARGET_ORDER:
    h2.append(
        r"& \multirow{2}{*}{Clean}"
        r" & \multicolumn{" + str(N_ADV) + r"}{c|}{Adversarial}"
        r" & \multirow{2}{*}{\textbf{Avg.}}"
    )
A("  ".join(h2) + r" \\")

# Header row 3: individual pert labels
h3 = [""]
adv_heads = " & ".join(PERT_LABEL[p] for p in ADV_PERTS)
for tgt in TARGET_ORDER:
    h3.append(r"& & " + adv_heads + r" &")
A("  ".join(h3) + r" \\")
A(r"\midrule")

# Surrogate blocks
for bi, surr in enumerate(SURROGATE_ORDER):
    A(r"\multicolumn{" + str(N_TOTAL) + r"}{l}{"
      r"\textit{Surrogate: \textbf{" + SURR_LABEL[surr] + r"}}}  \\")
    A(r"\addlinespace[1pt]")

    surr_adv_vals = {tgt: [] for tgt in TARGET_ORDER}
    for ds in DATASETS:
        row_cells = [r"\hspace{4pt}" + DS_LABEL[ds]]
        for tgt in TARGET_ORDER:
            row_cells.append(fmt(get_clean(tgt, ds)))
            ds_adv = []
            for p in ADV_PERTS:
                v = get(surr, p, tgt, ds)
                row_cells.append(fmt(v))
                if not np.isnan(v):
                    ds_adv.append(v)
                    surr_adv_vals[tgt].append(v)
            row_cells.append(fmt(nanmean(ds_adv)))
        A(" & ".join(row_cells) + r" \\")

    A(r"\addlinespace[1pt]")
    avg_cells = [r"\hspace{4pt}\textit{Avg.}"]
    for tgt in TARGET_ORDER:
        avg_cells.append(fmt(nanmean([get_clean(tgt, ds) for ds in DATASETS]), bold=True))
        for p in ADV_PERTS:
            avg_cells.append(fmt(nanmean([get(surr, p, tgt, ds) for ds in DATASETS]), bold=True))
        avg_cells.append(fmt(nanmean(surr_adv_vals[tgt]), bold=True))
    A(" & ".join(avg_cells) + r" \\")

    if bi < len(SURROGATE_ORDER) - 1:
        A(r"\midrule")

# Global average
A(r"\midrule")
glob = [r"\textbf{Global avg.}"]
for tgt in TARGET_ORDER:
    glob.append(fmt(nanmean([get_clean(tgt, ds) for ds in DATASETS]), bold=True))
    all_adv = []
    for p in ADV_PERTS:
        col = [get(surr, p, tgt, ds) for surr in SURROGATE_ORDER for ds in DATASETS]
        glob.append(fmt(nanmean(col), bold=True))
        all_adv.extend([v for v in col if not np.isnan(v)])
    glob.append(fmt(nanmean(all_adv), bold=True))
A(" & ".join(glob) + r" \\")

A(r"\bottomrule")
A(r"\end{tabular}")
A(r"}")
A(r"\end{table*}")

tex1 = OUT / "transfer_table.tex"
tex1.write_text("\n".join(lines))
print(f"Saved → {tex1}")

# ── CSV (adversarial only) ────────────────────────────────────────────────────
csv_rows = []
for surr in SURROGATE_ORDER:
    for ds in DATASETS:
        for tgt in TARGET_ORDER:
            row = {"surrogate": SURR_LABEL[surr], "dataset": DS_LABEL[ds],
                   "target": TARGET_LABELS[tgt], "clean": get_clean(tgt, ds)}
            for p in ADV_PERTS:
                row[p] = get(surr, p, tgt, ds)
            adv_vals = [row[p] for p in ADV_PERTS if not np.isnan(row[p])]
            row["avg_robust"] = nanmean(adv_vals)
            csv_rows.append(row)

csv1 = OUT / "transfer_table.csv"
with open(csv1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["surrogate","dataset","target","clean"]
                                     + ADV_PERTS + ["avg_robust"])
    w.writeheader()
    for row in csv_rows:
        w.writerow({k: ("" if isinstance(v, float) and np.isnan(v)
                        else round(v*100, 2) if isinstance(v, float) else v)
                    for k, v in row.items()})
print(f"Saved → {csv1}  ({len(csv_rows)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Common corruptions (surrogate-independent, one value per target×dataset)
# Rows: datasets + avg
# Columns: Dataset || Gemini: Clean | Sev@3 || GPT-4o: Clean | Sev@3
# ══════════════════════════════════════════════════════════════════════════════
# col spec: l || r r || r r
COL_COMMON = r"l || r r || r r"

lines2 = []
B = lambda s: lines2.append(s)

B(r"% Requires: booktabs, multirow")
B(r"\begin{table}[t]")
B(r"\centering")
B(r"\setlength{\tabcolsep}{5pt}")
B(r"\renewcommand{\arraystretch}{1.08}")
B(r"\small")
B(r"\caption{%")
B(r"  Robustness to common corruptions (severity 3).")
B(r"  Results are surrogate-independent: the same corrupted images")
B(r"  are used regardless of attack surrogate.")
B(r"  \emph{Clean} accuracy repeated for reference.")
B(r"}")
B(r"\label{tab:common}")
B(r"\begin{tabular}{" + COL_COMMON + r"}")
B(r"\toprule")

# Header row 1: target super-groups
B(r"\multirow{2}{*}{\textbf{Dataset}}"
  r" & \multicolumn{2}{c||}{\textbf{" + TARGET_LABELS["google_nothink"] + r"}}"
  r" & \multicolumn{2}{c}{\textbf{"   + TARGET_LABELS["openai"]          + r"}}"
  r" \\")

# Header row 2: Clean | Sev@3 per target
B(r" & Clean & Sev.$^3$ & Clean & Sev.$^3$ \\")
B(r"\midrule")

for ds in DATASETS:
    cells = [DS_LABEL[ds]]
    for tgt in TARGET_ORDER:
        cells.append(fmt(get_clean(tgt, ds)))
        cells.append(fmt(get("__common__", "sev@3", tgt, ds)))
    B(" & ".join(cells) + r" \\")

B(r"\midrule")
avg_cells2 = [r"\textbf{Average}"]
for tgt in TARGET_ORDER:
    avg_cells2.append(fmt(nanmean([get_clean(tgt, ds) for ds in DATASETS]), bold=True))
    avg_cells2.append(fmt(nanmean([get("__common__", "sev@3", tgt, ds)
                                   for ds in DATASETS]), bold=True))
B(" & ".join(avg_cells2) + r" \\")

B(r"\bottomrule")
B(r"\end{tabular}")
B(r"\end{table}")

tex2 = OUT / "common_table.tex"
tex2.write_text("\n".join(lines2))
print(f"Saved → {tex2}")