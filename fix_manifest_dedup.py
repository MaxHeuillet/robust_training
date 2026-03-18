#!/usr/bin/env python3
"""
fix_manifest_dedup.py — Deduplicate manifest and reset failed OpenAI entries for retry.

Run once:
    python fix_manifest_dedup.py llm_classification_results/batch_manifest__all_datasets__test_v1.json
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "llm_classification_results/batch_manifest__all_datasets__test_v1.json")

manifest = json.loads(path.read_text())
print(f"Before: {len(manifest)} entries")

# Known failed OpenAI batch IDs (token_limit_exceeded, total=0)
failed_openai_ids = {
    "batch_69ab7d083bb48190bab7b0f19667408e",  # stanford_cars
    "batch_69ab7c6f6524819091fc768127bdb27c",  # oxford-iiit-pet
    "batch_69ab7be0896481909a65eecc90328e60",  # fgvc-aircraft-2013b
    "batch_69ac5c7fec3c819094742ccb73a9494e",  # oxford-iiit-pet retry
    "batch_69ac5d041b4481909b606e7714b8d6ac",  # stanford_cars retry
}

# For each run_name, keep the best entry:
# priority: retrieved > submitted/pending with valid id > everything else
def entry_score(e):
    if e.get("status") == "retrieved":
        return 3
    bid = e.get("batch_id")
    if bid and bid not in failed_openai_ids:
        return 2
    return 0  # failed, null, or known-bad id → will be retried

seen: dict[str, dict] = {}
for entry in manifest:
    rn = entry["run_name"]
    if rn not in seen or entry_score(entry) > entry_score(seen[rn]):
        seen[rn] = entry

# Reset any entry with a known-bad batch_id so launcher retries it
clean = []
for entry in seen.values():
    if entry.get("batch_id") in failed_openai_ids:
        print(f"  Resetting {entry['run_name']} (failed batch {entry['batch_id']})")
        entry["batch_id"] = None
        entry["status"]   = "failed"
    clean.append(entry)

# Sort for readability
order = ["caltech101", "fgvc-aircraft-2013b", "flowers-102",
         "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset"]
key_order = ["google_think", "google_nothink", "anthropic", "openai"]
clean.sort(key=lambda e: (order.index(e["dataset"]) if e["dataset"] in order else 99,
                          key_order.index(e["key"]) if e["key"] in key_order else 99))

print(f"After:  {len(clean)} entries")
for e in clean:
    print(f"  {e['dataset']:<30} {e['key']:<16} {e['status']:<12} {e['batch_id'] or 'null'}")

path.write_text(json.dumps(clean, indent=2))
print(f"\nSaved to {path}")
print("\nNow retry failed jobs with:")
print(f"  python launch_batches.py --all_datasets --name test_v1")