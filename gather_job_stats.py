#!/usr/bin/env python3
"""
gather_job_stats.py
Cross-references the full expected (size, config, seed, dataset) matrix for the
Base/Small/Tiny FFT-50 multiseed reviewer-response runs against what's actually
on disk (trained state dicts, RobustGenBench eval CSVs) and what's currently
live in the SLURM queue, so preliminary results / gaps can be inspected without
waiting for every job to finish.

Usage:
    python3 gather_job_stats.py                # full report to stdout
    python3 gather_job_stats.py --csv out.csv  # also write a per-row CSV

No non-stdlib dependencies - runs fine on the login node with system python3.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent
TRAINED_STATEDICTS_PATH = Path(os.path.expanduser("~/links/scratch/mheuill/robust_training/trained_statedicts"))
EVAL_PATH = REPO / "robustgenbench_eval"

SEEDS = [1, 2, 3]
DATASETS = [
    "flowers-102",
    "stanford_cars",
    "oxford-iiit-pet",
    "caltech101",
    "fgvc-aircraft-2013b",
    "uc-merced-land-use-dataset",
]

# size -> [(medal, backbone, loss, prnm_base), ...]
SIZES = {
    "Base": [
        ("Gold", "convnext_base.fb_in22k", "TRADES_v2", "convnext_base_fb_in22k_TRADES_v2"),
        ("Silver", "coatnet_2_rw_224.sw_in12k_ft_in1k", "TRADES_v2", "coatnet_2_rw_224_sw_in12k_ft_in1k_TRADES_v2"),
        ("Bronze", "coatnet_2_rw_224.sw_in12k", "TRADES_v2", "coatnet_2_rw_224_sw_in12k_TRADES_v2"),
    ],
    "Small": [
        ("Gold", "convnext_tiny.fb_in22k_ft_in1k", "TRADES_v2", "convnext_tiny_fb_in22k_ft_in1k_TRADES_v2"),
        ("Silver", "convnext_tiny.fb_in1k", "TRADES_v2", "convnext_tiny_fb_in1k_TRADES_v2"),
        ("Bronze", "convnext_tiny.fb_in22k", "TRADES_v2", "convnext_tiny_fb_in22k_TRADES_v2"),
    ],
    "Tiny": [
        ("Gold", "coat_tiny.in1k", "TRADES_v2", "coat_tiny_in1k_TRADES_v2"),
        ("Silver", "edgenext_small.usi_in1k", "TRADES_v2", "edgenext_small_usi_in1k_TRADES_v2"),
        ("Bronze", "edgenext_small.usi_in1k", "CLASSIC_AT", "edgenext_small_usi_in1k_CLASSIC_AT"),
    ],
}


def get_queue_combos():
    """Returns {(backbone, loss, seed, dataset): (job_id, state)} for every
    job2/job3 currently known to squeue for this user, parsed from each job's
    SubmitLine (which carries the original --export values)."""
    combos = {}
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j %T %r"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception as e:
        print(f"warning: squeue failed ({e}), skipping live-queue cross-reference", file=sys.stderr)
        return combos

    jobs = [line.split(None, 3) for line in out.splitlines() if line.strip()]
    jobs = [(jid, name, state, reason) for jid, name, state, reason in jobs
            if name in ("job2_train_multiseed.sh", "job3_test_robustgenbench_multiseed.sh")]

    for jid, _name, state, reason in jobs:
        if state == "PENDING" and reason == "JobHeldUser":
            state = "HELD"
        try:
            info = subprocess.run(
                ["scontrol", "show", "job", jid], capture_output=True, text=True, check=True
            ).stdout
        except Exception:
            continue
        m = re.search(r"SubmitLine=(.*)", info)
        if not m:
            continue
        line = m.group(1)

        def grab(key):
            mm = re.search(rf"{key}=([^\s,]+)", line)
            return mm.group(1) if mm else None

        bckbn = grab("BCKBN")
        data = grab("DATA")
        seed = grab("SEED")
        loss = grab("LOSS")
        if bckbn and data and seed and loss:
            combos[(bckbn, loss, int(seed), data)] = (jid, state)
    return combos


def read_clean_accuracy(csv_path):
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("threat_model") == "clean":
                    return float(row["accuracy"])
    except Exception:
        pass
    return None


def main():
    write_csv = None
    if "--csv" in sys.argv:
        write_csv = sys.argv[sys.argv.index("--csv") + 1]

    queue_combos = get_queue_combos()

    rows = []
    for size, configs in SIZES.items():
        for medal, backbone, loss, prnm_base in configs:
            for seed in SEEDS:
                prnm = f"{prnm_base}_seed{seed}"
                for dataset in DATASETS:
                    statedict = TRAINED_STATEDICTS_PATH / prnm / f"{backbone}_{dataset}_{loss}.pt"
                    eval_csv = EVAL_PATH / prnm / f"{backbone}__{dataset}__{loss}__robustgenbench_results.csv"

                    trained = statedict.exists()
                    evaluated = eval_csv.exists()
                    clean_acc = read_clean_accuracy(eval_csv) if evaluated else None

                    if evaluated:
                        status = "EVALUATED"
                    elif trained:
                        status = "TRAINED_ONLY"
                    elif (backbone, loss, seed, dataset) in queue_combos:
                        jid, state = queue_combos[(backbone, loss, seed, dataset)]
                        status = f"QUEUED({state})"
                    else:
                        status = "MISSING"

                    rows.append(dict(
                        size=size, medal=medal, backbone=backbone, loss=loss,
                        seed=seed, dataset=dataset, prnm=prnm, status=status,
                        clean_accuracy=clean_acc,
                    ))

    # ---- console report ----
    total = len(rows)
    by_status = {}
    for r in rows:
        key = r["status"].split("(")[0]
        by_status[key] = by_status.get(key, 0) + 1

    print("=" * 78)
    print(f"Job status summary ({total} expected (config, seed, dataset) combos)")
    print("=" * 78)
    for k in ("EVALUATED", "TRAINED_ONLY", "QUEUED", "MISSING"):
        n = by_status.get(k, 0)
        print(f"  {k:<14} {n:>4}  ({100*n/total:5.1f}%)")
    print()

    for size in SIZES:
        size_rows = [r for r in rows if r["size"] == size]
        print("-" * 78)
        print(f"{size}")
        print("-" * 78)
        for medal, backbone, loss, prnm_base in SIZES[size]:
            medal_rows = [r for r in size_rows if r["medal"] == medal]
            counts = {}
            for r in medal_rows:
                key = r["status"].split("(")[0]
                counts[key] = counts.get(key, 0) + 1
            accs = [r["clean_accuracy"] for r in medal_rows if r["clean_accuracy"] is not None]
            acc_str = f"mean clean acc={mean(accs):.3f} (n={len(accs)})" if accs else "no results yet"
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  {medal:<7} {backbone:<38} {loss:<12} [{counts_str}]  {acc_str}")

            missing = [r for r in medal_rows if r["status"] == "MISSING"]
            if missing:
                miss_desc = ", ".join(f"seed{r['seed']}/{r['dataset']}" for r in missing)
                print(f"          MISSING: {miss_desc}")
        print()

    if write_csv:
        with open(write_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Full per-row status written to {write_csv}")


if __name__ == "__main__":
    main()
