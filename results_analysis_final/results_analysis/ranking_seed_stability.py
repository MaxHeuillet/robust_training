"""
ranking_seed_stability.py
==========================
Checks whether the Gold/Silver/Bronze ranking (as originally assigned via
Borda ranking against the full candidate pool, see process_database.py and
paper_table3.ipynb) holds when the sum-score of clean/Linf/L2/L1/common
accuracy across all 6 RobustGenBench datasets is recomputed, per seed, for
just the 3 medal winners across 3 newly-trained, independent seeds
(results/{prnm}_seed{N}/*.pkl).

For each medal, the sum-score is computed separately for each of the 3
seeds (accuracy summed across 6 datasets x 5 metrics), then reported as
mean +/- sample standard deviation across the 3 per-seed sums -- this
directly shows seed-to-seed variance rather than pooling everything into a
single number.

Gold/Silver/Bronze are treated as FIXED labels here (already assigned by the
original Borda-over-full-pool ranking) -- this script does not re-derive
them, it only checks whether an independent, simpler metric (raw sum-score
over just these 3) agrees with that fixed label order once real seed
variance is folded in.

Usage:
    python ranking_seed_stability.py

Run from the repo root (paths are relative to robust_training/).
"""

import pickle
import os
import statistics

RESULTS_DIR = "results"
DATASETS = ["flowers-102", "stanford_cars", "oxford-iiit-pet", "caltech101",
            "fgvc-aircraft-2013b", "uc-merced-land-use-dataset"]
METRICS = ["clean", "Linf", "L2", "L1", "common"]
SEEDS = [1, 2, 3]

# medal -> backbone, or (backbone, loss) if loss differs from the experiment default
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


def load_seed(backbone, loss, seed, ds):
    prnm = backbone.replace(".", "_") + f"_{loss}_seed{seed}"
    path = f"{RESULTS_DIR}/{prnm}/{backbone}_{ds}_{loss}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def per_seed_sum(backbone, loss, seed):
    """Sum of accuracy across 6 datasets x 5 metrics for one seed."""
    total = 0.0
    for ds in DATASETS:
        d = load_seed(backbone, loss, seed, ds)
        if d:
            for m in METRICS:
                v = d.get(f"{m}_acc")
                if v is not None:
                    total += v
    return total


def medal_coverage(backbone, loss):
    """How many of the 18 (seed x dataset) new-seed pkls exist for this medal."""
    n = 0
    for seed in SEEDS:
        for ds in DATASETS:
            if load_seed(backbone, loss, seed, ds) is not None:
                n += 1
    return n


def run(experiments=EXPERIMENTS, require_full_coverage=True):
    for exp_name, medals in experiments.items():
        specs = {m: (v if isinstance(v, tuple) else (v, "TRADES_v2")) for m, v in medals.items()}

        coverage = {m: medal_coverage(bb, ls) for m, (bb, ls) in specs.items()}
        if require_full_coverage and any(c < len(SEEDS) * len(DATASETS) for c in coverage.values()):
            print(f"{exp_name}: SKIPPED (incomplete new-seed coverage: {coverage}, need {len(SEEDS)*len(DATASETS)} each)")
            continue

        print(f"\n{'='*70}\n{exp_name}\n{'='*70}")
        stats = {}
        for medal, (backbone, loss) in specs.items():
            sums = [per_seed_sum(backbone, loss, s) for s in SEEDS]
            stats[medal] = (statistics.mean(sums), statistics.stdev(sums), sums)

        order = sorted(stats, key=lambda m: stats[m][0], reverse=True)
        for medal in ("Gold", "Silver", "Bronze"):
            mean, std, sums = stats[medal]
            per_seed_str = ", ".join(f"seed{s}={v:.4f}" for s, v in zip(SEEDS, sums))
            print(f"    {medal:<7} mean={mean:.4f}  std={std:.4f}  ({per_seed_str})")
        print(f"    -> order: {' > '.join(order)}"
              f"{'  [MATCHES Gold>Silver>Bronze]' if order == ['Gold','Silver','Bronze'] else '  [DOES NOT MATCH]'}")


if __name__ == "__main__":
    run()
