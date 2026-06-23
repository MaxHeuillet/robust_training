#!/usr/bin/env python3
"""
summarize_results.py — Read all batch manifests, load predictions.jsonl
for each retrieved run, compute accuracy, and print a full summary table.
Also identifies missing / not-yet-retrieved entries.
"""

import json
from pathlib import Path

BASE = Path("llm_classification_results")

MANIFESTS = [
    "batch_manifest__all_datasets__adv_siglip2_linf8.json",
    "batch_manifest__all_datasets__adv_siglip2_linf30.json",
]

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

DATASET_SHORT = {
    "caltech101":                 "Caltech",
    "fgvc-aircraft-2013b":        "FGVC",
    "flowers-102":                "Flowers",
    "oxford-iiit-pet":            "Ox.Pet",
    "stanford_cars":              "S.Cars",
    "uc-merced-land-use-dataset": "UCMerced",
}

# ---------------------------------------------------------------------------

def dataset_key(run_name: str) -> str:
    """Extract bare dataset name from run_name."""
    for ds in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(ds):
            return ds
    return run_name.split("__")[0]


def load_predictions(run_name: str) -> tuple[int, int] | None:
    """Returns (correct, total) or None if predictions not available."""
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
    return correct, total if total > 0 else None


def merge_complement(main_run: str, comp_run: str) -> tuple[int, int] | None:
    """Merge main + complement predictions, return (correct, total)."""
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
    return correct, total if total > 0 else None


# ---------------------------------------------------------------------------
# Load all manifests

all_entries = []   # list of dicts with experiment, dataset, model_key, status, acc

for mname in MANIFESTS:
    mp = BASE / mname
    if not mp.exists():
        print(f"  ⚠ Manifest not found: {mname}")
        continue
    manifest = json.loads(mp.read_text())
    exp = mname.replace("batch_manifest__all_datasets__", "").replace(".json", "")

    # Group by (dataset, key) to detect complement pairs
    groups: dict[tuple, list] = {}
    for entry in manifest:
        ds  = dataset_key(entry["run_name"])
        key = entry["key"]
        groups.setdefault((ds, key), []).append(entry)

    for (ds, key), entries in groups.items():
        # Separate main from complement
        main_entries = [e for e in entries if "__complement" not in e["run_name"]]
        comp_entries = [e for e in entries if "__complement" in e["run_name"]]

        # Prefer retrieved entries over failed ones
        main_entries_ok = [e for e in main_entries if e["status"] != "failed"]
        main = main_entries_ok[0] if main_entries_ok else (main_entries[0] if main_entries else entries[0])
        comp  = comp_entries[0] if comp_entries else None
        status = main["status"]

        acc = None
        if status == "retrieved":
            if comp and comp["status"] == "retrieved":
                result = merge_complement(main["run_name"], comp["run_name"])
            else:
                result = load_predictions(main["run_name"])
            if result:
                correct, total = result
                acc = correct / total if total > 0 else None

        all_entries.append({
            "experiment": exp,
            "dataset":    ds,
            "model_key":  key,
            "status":     status,
            "acc":        acc,
            "run_name":   main["run_name"],
        })

# ---------------------------------------------------------------------------
# Print results table

experiments = sorted({e["experiment"] for e in all_entries})
model_keys  = sorted({e["model_key"]  for e in all_entries})

print(f"\n{'='*90}")
print(f"  FULL RESULTS — accuracy by experiment × dataset × model")
print(f"{'='*90}\n")

for exp in experiments:
    exp_entries = [e for e in all_entries if e["experiment"] == exp]
    keys_present = sorted({e["model_key"] for e in exp_entries})

    col_w = 14
    header = f"{'Dataset':<14}" + "".join(f"{k:>{col_w}}" for k in keys_present)
    print(f"  [{exp}]")
    print(f"  {header}")
    print(f"  {'─'*(14 + col_w*len(keys_present))}")

    for ds in DATASETS:
        row = f"  {DATASET_SHORT[ds]:<14}"
        for k in keys_present:
            match = [e for e in exp_entries if e["dataset"] == ds and e["model_key"] == k]
            if not match:
                row += f"{'—':>{col_w}}"
            else:
                e = match[0]
                if e["acc"] is not None:
                    row += f"{e['acc']*100:>{col_w-1}.1f}%"
                elif e["status"] == "retrieved":
                    row += f"{'(no preds)':>{col_w}}"
                elif e["status"] == "submitted":
                    row += f"{'(pending)':>{col_w}}"
                elif e["status"] == "failed":
                    row += f"{'(failed)':>{col_w}}"
                else:
                    row += f"{e['status']:>{col_w}}"
        print(row)
    print()

# ---------------------------------------------------------------------------
# Missing / pending entries

print(f"{'='*90}")
print(f"  MISSING / PENDING ENTRIES")
print(f"{'='*90}")

missing = [e for e in all_entries if e["acc"] is None]
if not missing:
    print("  All entries retrieved and have predictions.\n")
else:
    for e in sorted(missing, key=lambda x: (x["experiment"], x["dataset"], x["model_key"])):
        print(f"  [{e['status']:>10}]  {e['experiment']:<35}  {DATASET_SHORT[e['dataset']]:<10}  {e['model_key']}")
    print()

