#!/usr/bin/env python3
"""
compare_clean_vs_adv.py — Compare LLM accuracy on clean vs adversarial images
for the overlapping subset of indices.
"""

import json
from pathlib import Path

CLEAN_PATH = Path("/Users/maximeheuillet/Desktop/robust_training/llm_classification_results/flowers-102__openai__test_v1/predictions.jsonl")
ADV_PATH   = Path("/Users/maximeheuillet/Desktop/robust_training/llm_classification_results/flowers-102__adv_linf30__zeroshot_clip__openai/predictions.jsonl")

def load_predictions(path: Path) -> dict:
    """Load predictions indexed by image index."""
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[rec["index"]] = rec
    return records

clean_preds = load_predictions(CLEAN_PATH)
adv_preds   = load_predictions(ADV_PATH)

# Only compare on the overlapping subset
shared_indices = sorted(set(clean_preds.keys()) & set(adv_preds.keys()))
n = len(shared_indices)

print(f"\n{'='*55}")
print(f"  Overlapping subset : {n} images")
print(f"{'='*55}")

clean_correct = sum(1 for i in shared_indices if clean_preds[i]["correct"])
adv_correct   = sum(1 for i in shared_indices if adv_preds[i]["correct"])

clean_acc = clean_correct / n
adv_acc   = adv_correct   / n
delta     = adv_acc - clean_acc

print(f"  Clean accuracy     : {clean_acc:.4f}  ({clean_correct}/{n})")
print(f"  Adv accuracy       : {adv_acc:.4f}  ({adv_correct}/{n})")
print(f"  Delta              : {delta:+.4f}  ({'↓ degraded' if delta < 0 else '↑ improved' if delta > 0 else '= no change'})")
print(f"  Attack success     : {1 - adv_acc:.4f}  (fraction of clean-correct flipped)")
print(f"{'='*55}\n")

# Breakdown: what happened to each clean-correct prediction
flipped   = 0  # clean correct, adv wrong
held      = 0  # clean correct, adv also correct
recovered = 0  # clean wrong,   adv correct
both_wrong= 0  # clean wrong,   adv wrong

for i in shared_indices:
    c = clean_preds[i]["correct"]
    a = adv_preds[i]["correct"]
    if c and not a:     flipped    += 1
    elif c and a:       held       += 1
    elif not c and a:   recovered  += 1
    else:               both_wrong += 1

print(f"  Breakdown of {n} images:")
print(f"  ✓→✗  Flipped by attack     : {flipped}  ({flipped/n:.2%})")
print(f"  ✓→✓  Held against attack   : {held}     ({held/n:.2%})")
print(f"  ✗→✓  Recovered by noise    : {recovered} ({recovered/n:.2%})")
print(f"  ✗→✗  Wrong in both cases   : {both_wrong} ({both_wrong/n:.2%})")
print(f"{'='*55}\n")