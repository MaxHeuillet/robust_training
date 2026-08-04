"""
ranking_bootstrap_stability.py
================================
Companion to ranking_seed_stability.py, but isolates a different source of
noise: instead of varying the *training* seed, this resamples the *test
set* itself (nonparametric bootstrap, with replacement) to see how much the
Gold/Silver/Bronze sum-score ranking wobbles from finite-sample evaluation
noise alone, holding the trained model fixed.

Requires the per-observation predictions CSVs written by
job3_test_whitebox_multiseed.sh / distributed_experiment_final.py's
write_whitebox_predictions_csv() (results/{prnm}/predictions/*.csv). One
CSV per (medal, dataset, threat_label); each row is one test observation
with a 0/1 `correct` column.

For each of B bootstrap iterations:
  - independently resample (with replacement) each (medal, dataset, metric)
    observation set and recompute accuracy
  - sum the resampled accuracies across all datasets/metrics -> one total
    score per medal for that iteration
  - record whether Gold > Silver > Bronze holds for that iteration

Reports: % of iterations preserving the Gold>Silver>Bronze order, and a
percentile CI per medal's total score.

Usage:
    python ranking_bootstrap_stability.py [--seed N] [--iters B]

Run from the repo root (paths are relative to robust_training/).
"""

import argparse
import csv
import os
import random
from collections import defaultdict

RESULTS_DIR = "results"
DATASETS = ["flowers-102", "stanford_cars", "oxford-iiit-pet", "caltech101",
            "fgvc-aircraft-2013b", "uc-merced-land-use-dataset"]
LABELS = ["clean", "Linf", "L2", "L1", "common"]  # matches predictions CSV suffix

EXPERIMENTS = {
    "Base": {
        "Gold":   "convnext_base.fb_in22k",
        "Silver": "coatnet_2_rw_224.sw_in12k_ft_in1k",
        "Bronze": "coatnet_2_rw_224.sw_in12k",
    },
    "Small": {
        "Gold":   "convnext_tiny.fb_in22k_ft_in1k",
        "Silver": "convnext_tiny.fb_in1k",
        "Bronze": "convnext_tiny.fb_in22k",
    },
    "Tiny": {
        "Gold":   ("coat_tiny.in1k", "TRADES_v2"),
        "Silver": ("edgenext_small.usi_in1k", "TRADES_v2"),
        "Bronze": ("edgenext_small.usi_in1k", "CLASSIC_AT"),
    },
}


def load_correct_flags(backbone, loss, seed, ds, label):
    """Returns list of 0/1 ints from the predictions CSV, or None if missing."""
    prnm = backbone.replace(".", "_") + f"_{loss}_seed{seed}"
    exp_id = f"{backbone}_{ds}_{loss}"
    path = f"{RESULTS_DIR}/{prnm}/predictions/{exp_id}__{label}__predictions.csv"
    if not os.path.exists(path):
        return None
    flags = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            flags.append(int(row["correct"]))
    return flags if flags else None


def bootstrap_accuracy_samples(flags, n_iters, rng):
    """Returns a list of n_iters bootstrap-resampled accuracy estimates."""
    n = len(flags)
    out = []
    for _ in range(n_iters):
        resample = [flags[rng.randrange(n)] for _ in range(n)]
        out.append(sum(resample) / n)
    return out


def percentile(sorted_vals, p):
    idx = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def run(experiments=EXPERIMENTS, seed=1, n_iters=1000, rng_seed=0):
    rng = random.Random(rng_seed)

    for exp_name, medals in experiments.items():
        specs = {m: (v if isinstance(v, tuple) else (v, "TRADES_v2")) for m, v in medals.items()}

        # gather all available (dataset, label) flag-lists per medal
        per_medal_flags = {}
        missing = []
        for medal, (backbone, loss) in specs.items():
            per_medal_flags[medal] = {}
            for ds in DATASETS:
                for label in LABELS:
                    flags = load_correct_flags(backbone, loss, seed, ds, label)
                    if flags is not None:
                        per_medal_flags[medal][(ds, label)] = flags
                    elif label != "common":  # common is expected to be missing sometimes
                        missing.append((medal, ds, label))

        n_points = {m: len(v) for m, v in per_medal_flags.items()}
        if any(n == 0 for n in n_points.values()):
            print(f"{exp_name}: SKIPPED (no predictions CSVs found for seed{seed}: {n_points})")
            continue

        # only use (dataset,label) keys present for ALL 3 medals, for a fair comparison
        common_keys = sorted(set.intersection(*(set(v.keys()) for v in per_medal_flags.values())))
        if not common_keys:
            print(f"{exp_name}: SKIPPED (no (dataset,label) keys shared across all 3 medals for seed{seed})")
            continue

        print(f"\n{'='*70}\n{exp_name}  (seed{seed}, {n_iters} bootstrap iterations, "
              f"{len(common_keys)} matched dataset/threat combos)\n{'='*70}")

        # bootstrap: independently resample each (medal, dataset, label) observation set
        per_medal_iter_totals = {m: [0.0] * n_iters for m in specs}
        for medal in specs:
            for key in common_keys:
                flags = per_medal_flags[medal][key]
                samples = bootstrap_accuracy_samples(flags, n_iters, rng)
                for i, v in enumerate(samples):
                    per_medal_iter_totals[medal][i] += v

        order_counts = defaultdict(int)
        for i in range(n_iters):
            order = tuple(sorted(specs, key=lambda m: per_medal_iter_totals[m][i], reverse=True))
            order_counts[order] += 1

        print(f"  order frequency across {n_iters} bootstrap iterations:")
        for order, count in sorted(order_counts.items(), key=lambda kv: -kv[1]):
            pct = 100 * count / n_iters
            tag = "  [MATCHES Gold>Silver>Bronze]" if order == ("Gold", "Silver", "Bronze") else ""
            print(f"    {' > '.join(order):<30} {pct:5.1f}%  ({count}/{n_iters}){tag}")

        gold_top_pct = 100 * sum(c for o, c in order_counts.items() if o[0] == "Gold") / n_iters
        print(f"  P(Gold ranks 1st) = {gold_top_pct:.1f}%")

        for medal in ("Gold", "Silver", "Bronze"):
            vals = sorted(per_medal_iter_totals[medal])
            lo, hi = percentile(vals, 0.025), percentile(vals, 0.975)
            mean = sum(vals) / len(vals)
            print(f"    {medal:<7} mean_total={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="which training seed's predictions to bootstrap")
    parser.add_argument("--iters", type=int, default=1000, help="number of bootstrap iterations")
    args = parser.parse_args()
    run(seed=args.seed, n_iters=args.iters)
