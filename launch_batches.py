#!/usr/bin/env python3
"""
launch_batches.py — Submit batch classification jobs across models and datasets.

Usage:
    # Single dataset (prototyping)
    python launch_batches.py --datasets flowers-102

    # All datasets
    python launch_batches.py --all_datasets

    # With an experiment name
    python launch_batches.py --datasets flowers-102 --name baseline_v1

    # Custom paths
    python launch_batches.py --datasets flowers-102 --name test \\
        --data_root ~/data_processed \\
        --class_names_dir ~/data_processed/class_names \\
        --output_dir ./results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    # "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

MODELS = [
    {"key": "google_think",   "provider": "google",    "model": "gemini-3-flash-preview-think"},
    {"key": "google_nothink", "provider": "google",    "model": "gemini-3-flash-preview-nothink"},
    {"key": "anthropic",      "provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"key": "openai",         "provider": "openai",    "model": "gpt-4o-mini"},
]

# ---------------------------------------------------------------------------

def run_name_for(dataset: str, key: str, experiment: str | None) -> str:
    parts = [dataset, key]
    if experiment:
        parts.append(experiment)
    return "__".join(parts)


def submit_job(
    provider: str,
    model: str,
    dataset: str,
    run_name: str,
    data_root: str,
    class_names_dir: str,
    output_dir: str,
    split: str,
    max_samples: int | None,
) -> str | None:
    """Call llm_classify.py --batch and return the batch ID, or None on failure."""
    cmd = [
        sys.executable, "llm_classify.py",
        "--batch",
        "--provider",        provider,
        "--model",           model,
        "--dataset",         dataset,
        "--split",           split,
        "--data_root",       data_root,
        "--class_names_dir", class_names_dir,
        "--output_dir",      output_dir,
        "--run_name",        run_name,
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠  llm_classify.py exited with code {result.returncode}")
        return None

    meta_path = Path(output_dir) / run_name / "batch_meta.json"
    if not meta_path.exists():
        print(f"  ⚠  batch_meta.json not found at {meta_path}")
        return None

    meta = json.loads(meta_path.read_text())
    return meta.get("batch_id") or meta.get("batch_name")


def main():
    p = argparse.ArgumentParser(description="Submit batch classification jobs")

    # Dataset selection
    ds_group = p.add_mutually_exclusive_group(required=True)
    ds_group.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, metavar="DATASET",
        help="One or more datasets to run (e.g. --datasets flowers-102 caltech101)",
    )
    ds_group.add_argument(
        "--all_datasets", action="store_true",
        help="Run on all datasets",
    )

    # Experiment name
    p.add_argument("--name", default="", metavar="EXPERIMENT_NAME",
                   help="Optional experiment name appended to every run name")

    # Paths
    p.add_argument("--data_root",       default="~/data_processed")
    p.add_argument("--class_names_dir", default="~/data_processed/class_names")
    p.add_argument("--output_dir",      default="./llm_classification_results")
    p.add_argument("--split",           default="test")
    p.add_argument("--max_samples",     type=int, default=None,
                   help="Cap samples per dataset (omit for full dataset)")

    args = p.parse_args()

    datasets        = ALL_DATASETS if args.all_datasets else args.datasets
    output_dir      = Path(os.path.expanduser(args.output_dir))
    data_root       = os.path.expanduser(args.data_root)
    class_names_dir = os.path.expanduser(args.class_names_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive manifest path from experiment name + datasets
    tag = "__".join(filter(None, [
        "_".join(d.replace("-", "") for d in datasets) if len(datasets) <= 3 else "all_datasets",
        args.name,
    ])) or "experiment"
    manifest_path = output_dir / f"batch_manifest__{tag}.json"

    # Load existing manifest so re-runs accumulate rather than overwrite
    if manifest_path.exists():
        manifest: list[dict] = json.loads(manifest_path.read_text())
        n = len(manifest)
        print(f"  (Found existing manifest with {n} entr{'y' if n == 1 else 'ies'} — will skip already-submitted jobs)")
    else:
        manifest: list[dict] = []

    # Index already-submitted entries by run_name for O(1) lookup
    already_submitted: dict[str, dict] = {
        e["run_name"]: e for e in manifest if e.get("batch_id")
    }

    print()
    print("=" * 60)
    print(f"  Datasets:   {', '.join(datasets)}")
    print(f"  Models:     {len(MODELS)}")
    print(f"  Experiment: {args.name or '<none>'}")
    print(f"  Manifest:   {manifest_path}")
    print("=" * 60)
    print()

    new_submitted = 0
    new_failed    = 0
    skipped       = 0

    for dataset in datasets:
        print(f"── Dataset: {dataset}")
        for entry in MODELS:
            key      = entry["key"]
            provider = entry["provider"]
            model    = entry["model"]
            run_name = run_name_for(dataset, key, args.name)

            print(f"   {key:<18}  provider={provider}  model={model}")
            print(f"   run_name: {run_name}")

            # Skip if already in the manifest with a valid batch_id
            if run_name in already_submitted:
                existing = already_submitted[run_name]
                print(f"   ✓ Already submitted — batch_id: {existing['batch_id']}  [skipped]")
                print()
                skipped += 1
                continue

            batch_id = submit_job(
                provider=provider,
                model=model,
                dataset=dataset,
                run_name=run_name,
                data_root=data_root,
                class_names_dir=class_names_dir,
                output_dir=str(output_dir),
                split=args.split,
                max_samples=args.max_samples,
            )

            status = "submitted" if batch_id else "failed"
            print(f"   → batch_id: {batch_id}  [{status}]")
            print()

            manifest.append({
                "dataset":    dataset,
                "key":        key,
                "provider":   provider,
                "model":      model,
                "run_name":   run_name,
                "batch_id":   batch_id,
                "status":     status,
                "experiment": args.name,
            })

            if status == "submitted":
                new_submitted += 1
            else:
                new_failed += 1

        # Write after every dataset so progress is never lost mid-run
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print("=" * 60)
    print(f"  Newly submitted: {new_submitted}   Failed: {new_failed}   Skipped: {skipped}")
    print(f"  Total in manifest: {len(manifest)}")
    print(f"  Manifest saved to: {manifest_path}")
    print()
    print("  When jobs are done, retrieve results with:")
    print(f"    python retrieve_batches.py {manifest_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()