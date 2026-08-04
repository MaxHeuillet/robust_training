"""
attention_trades_beta_analysis.py
===================================
Tests the claim (L231-232, supported by paper_figures/interaction_loss_function_model_type.png):
"TRADES and Classic AT yield equivalent outcomes (similar mean Borda score)
for attention-based architectures", and whether this still holds for TRADES
at beta=6.

Reasons in terms of sum-score (sum, not mean, of clean/Linf/L2/L1/common
accuracy across the 6 RobustGenBench datasets -- 30 values per backbone per
condition), not Borda rank -- Borda depends on the composition of the full
~38-backbone candidate pool and only 5 of the 8 "fully attention" backbones
were retrained at beta=6, so a Borda comparison would implicitly compare
unequal candidate sets. Sum-score is directly comparable across a MATCHED
set of backbones (same 5 backbones, same datasets, all 3 loss conditions),
which is what this script does.

Usage:
    python attention_trades_beta_analysis.py

Run from the repo root (paths relative to robust_training/).
"""

import pickle
import os

RESULTS_DIR = "results"
BASELINE_DIR = f"{RESULTS_DIR}/full_fine_tuning50"

DATASETS = ['flowers-102', 'stanford_cars', 'oxford-iiit-pet',
            'caltech101', 'fgvc-aircraft-2013b', 'uc-merced-land-use-dataset']
METRICS = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']

# The 5 (of 8) "fully attention" backbones that were retrained at TRADES beta=6.
# (The other 3 -- swin_base, eva02_tiny, deit_tiny -- were not retrained and
# are excluded here so every column is over the same matched set of backbones.)
BETA6_BACKBONES = [
    "deit_small_patch16_224.fb_in1k",
    "eva02_base_patch14_224.mim_in22k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_small_patch16_224.augreg_in1k",
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def sum_score(backbone, condition):
    """condition: 'TRADES_v2' | 'CLASSIC_AT' (baseline, beta=1) | 'beta6'.
    Returns (sum, n) -- sum of up to 30 accuracy values (6 datasets x 5 metrics)."""
    vals = []
    for ds in DATASETS:
        if condition == "beta6":
            prnm = backbone.replace(".", "_") + "_TRADES_beta6"
            d = load(f"{RESULTS_DIR}/{prnm}/{backbone}_{ds}_TRADES_v2.pkl")
        else:
            d = load(f"{BASELINE_DIR}/{backbone}__{ds}__{condition}.pkl")
        if d:
            for m in METRICS:
                v = d.get(m)
                if v is not None:
                    vals.append(v)
    return (sum(vals), len(vals)) if vals else (float("nan"), 0)


def run():
    print(f"{'Backbone':<42}{'TRADES(b1)':>14}{'ClassicAT':>14}{'TRADES(b6)':>14}   Best")
    print("-" * 100)

    col_totals = {"TRADES(b1)": [], "ClassicAT": [], "TRADES(b6)": []}
    wins = {"TRADES(b1)": 0, "ClassicAT": 0, "TRADES(b6)": 0}

    for bb in BETA6_BACKBONES:
        t1, n1 = sum_score(bb, "TRADES_v2")
        at, n2 = sum_score(bb, "CLASSIC_AT")
        t6, n3 = sum_score(bb, "beta6")
        col_totals["TRADES(b1)"].append(t1)
        col_totals["ClassicAT"].append(at)
        col_totals["TRADES(b6)"].append(t6)
        vals = {"TRADES(b1)": t1, "ClassicAT": at, "TRADES(b6)": t6}
        best = max(vals, key=vals.get)
        wins[best] += 1
        print(f"{bb:<42}{t1:>10.4f}(n={n1:>2}){at:>10.4f}(n={n2:>2}){t6:>10.4f}(n={n3:>2})   {best}")

    print("-" * 100)
    sums = {k: sum(v) for k, v in col_totals.items()}
    overall_best = max(sums, key=sums.get)
    print(f"{'TOTAL (matched, 5 backbones)':<42}{sums['TRADES(b1)']:>14.4f}"
          f"{sums['ClassicAT']:>14.4f}{sums['TRADES(b6)']:>14.4f}   {overall_best}")
    print(f"\nWin count across the 5 matched backbones: {wins}")


if __name__ == "__main__":
    run()
