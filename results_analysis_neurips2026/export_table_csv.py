#!/usr/bin/env python3
"""
export_table_csv.py

Exports the full transfer accuracy table to CSV.
Columns: dataset, target, surrogate, perturbation, accuracy
One row per (dataset × target × surrogate × perturbation) cell.
Clean accuracy included as perturbation="clean".
"""

import json
import csv
import numpy as np
from pathlib import Path

EXPERIMENT_META = {
    "test_v1":                    ("__clean__",      "clean"),
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

# ── Load ──────────────────────────────────────────────────────────────────────
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
        if key not in TARGET_LABELS: continue
        main_e  = [e for e in entries if "__complement" not in e["run_name"]]
        comp_e  = [e for e in entries if "__complement"     in e["run_name"]]
        main_ok = [e for e in main_e if e["status"] != "failed"]
        main    = main_ok[0] if main_ok else (main_e[0] if main_e else entries[0])
        comp    = comp_e[0] if comp_e else None
        if main["status"] != "retrieved": continue
        res = (merge_complement(main["run_name"], comp["run_name"])
               if comp and comp["status"] == "retrieved"
               else load_preds(main["run_name"]))
        if res:
            raw[(surr, pert, key, ds)] = res[0] / res[1]

# ── Write CSV ─────────────────────────────────────────────────────────────────
# For clean: one row per (dataset, target) — surrogate = "n/a"
# For sev@3: one row per (dataset, target) — surrogate = "n/a" (surrogate-independent)
# For adversarial: one row per (dataset, target, surrogate)

out = Path("./results_analysis_neurips2026/transfer_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
for (surr, pert, tgt, ds), acc in sorted(raw.items()):
    # clean: attribute to all surrogates would duplicate; keep as its own surrogate="clean"
    surrogate_col = (
        "n/a (clean)"   if surr == "__clean__"  else
        "n/a (common)"  if surr == "__common__" else
        surr
    )
    rows.append({
        "dataset":     ds,
        "target":      TARGET_LABELS[tgt],
        "surrogate":   surrogate_col,
        "perturbation": pert,
        "accuracy":    round(acc * 100, 2),
    })

# Sort: dataset → target → surrogate → perturbation
rows.sort(key=lambda r: (
    r["dataset"], r["target"], r["surrogate"], r["perturbation"]
))

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["dataset", "target", "surrogate", "perturbation", "accuracy"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved → {out}  ({len(rows)} rows)")

# ── Quick sanity preview ──────────────────────────────────────────────────────
print(f"\n{'dataset':<32} {'target':<28} {'surrogate':<20} {'perturbation':<12} {'accuracy':>8}")
print("-" * 108)
for r in rows[:12]:
    print(f"{r['dataset']:<32} {r['target']:<28} {r['surrogate']:<20} {r['perturbation']:<12} {r['accuracy']:>8.2f}")
print("  ...")