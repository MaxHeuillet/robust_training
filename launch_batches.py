#!/usr/bin/env python3
"""
launch_batches.py — Submit batch classification jobs across models and datasets.

Usage:
    # Clean datasets
    python launch_batches.py --all_datasets --name test_v1

    # Adversarial datasets
    python launch_batches.py --all_datasets --name adv_linf30 \
        --data_root /tmp/data/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard \
        --class_names_dir ~/data_processed/class_names

    # Complete missing predictions (e.g. after partial submission)
    python launch_batches.py --all_datasets --name adv_linf30 --complete_missing \
        --data_root /tmp/data/adversarial/... --class_names_dir ~/data_processed/class_names
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    # "caltech101",
    # "fgvc-aircraft-2013b",
    # "flowers-102",
    # "oxford-iiit-pet",
    "stanford_cars",
    # "uc-merced-land-use-dataset",
]

MODELS = [
    # {"key": "google_think",   "provider": "google",    "model": "gemini-3-flash-preview-think"},
    # {"key": "google_nothink", "provider": "google",    "model": "gemini-3-flash-preview-nothink"},
    # {"key": "anthropic",      "provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"key": "openai",         "provider": "openai",    "model": "gpt-4o-mini"},
]

# ---------------------------------------------------------------------------

def run_name_for(dataset: str, key: str, experiment: str | None) -> str:
    parts = [dataset, key]
    if experiment:
        parts.append(experiment)
    return "__".join(parts)


def resolve_dataset_name(dataset: str, data_root: str) -> str:
    """Match 'caltech101' to 'caltech101__zeroshot_..._processed.tar.zst'."""
    root = Path(data_root)
    matches = sorted(root.glob(f"{dataset}*_processed.tar.zst"))
    if not matches:
        return dataset
    if len(matches) > 1:
        print(f"  ⚠  Multiple archives for {dataset}: {[m.name for m in matches]} — using first")
    resolved = matches[0].name.replace("_processed.tar.zst", "")
    if resolved != dataset:
        print(f"  → Resolved {dataset!r} to {resolved!r}")
    return resolved


def get_done_indices(predictions_path: Path) -> set[int]:
    """Return set of image indices successfully processed (errors excluded so they get resubmitted)."""
    if not predictions_path.exists():
        return set()
    done = set()
    for line in predictions_path.read_text().splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                if not rec.get("error", False):
                    done.add(rec["index"])
            except Exception:
                pass
    return done


def get_total_indices(data_root: str, dataset: str, work_dir: str) -> list[int]:
    """Return all indices available in the test split labels.csv."""
    import tarfile
    try:
        import zstandard as zstd
    except ImportError:
        return []

    archive_path = Path(data_root) / f"{dataset}_processed.tar.zst"
    dest_dir     = Path(work_dir) / dataset

    if not (dest_dir / "test" / "labels.csv").exists():
        if archive_path.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            with open(archive_path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tar:
                        tar.extractall(path=dest_dir)

    csv_path = dest_dir / "test" / "labels.csv"
    if not csv_path.exists():
        return []

    with open(csv_path) as f:
        return list(range(len(list(csv.DictReader(f)))))


def submit_job(
    provider: str,
    model: str,
    dataset: str,
    run_name: str,
    data_root: str,
    class_names_dir: str,
    output_dir: str,
    split: str,
    indices_to_submit: Optional[list[int]] = None,
) -> str | None:
    """Call llm_classify.py --batch, optionally restricting to specific indices."""
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

    # If we have specific indices to submit, write them to a temp file
    # and pass as --indices_file (requires llm_classify.py support)
    # Otherwise fall back to max_samples
    if indices_to_submit is not None:
        cmd += ["--indices", ",".join(map(str, indices_to_submit))]

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


def submit_job_simple(
    provider: str,
    model: str,
    dataset: str,
    run_name: str,
    data_root: str,
    class_names_dir: str,
    output_dir: str,
    split: str,
    max_samples: int | None = None,
    indices: list[int] | None = None,
) -> str | None:
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
    if indices is not None:
        cmd += ["--indices", ",".join(map(str, indices))]

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

    ds_group = p.add_mutually_exclusive_group(required=True)
    ds_group.add_argument("--datasets",    nargs="+", metavar="DATASET")
    ds_group.add_argument("--all_datasets", action="store_true")

    p.add_argument("--name",            default="", metavar="EXPERIMENT_NAME")
    p.add_argument("--data_root",       default="~/data_processed")
    p.add_argument("--class_names_dir", default="~/data_processed/class_names")
    p.add_argument("--output_dir",      default="./llm_classification_results")
    p.add_argument("--split",           default="test")
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--work_dir",        default="/tmp/llm_classify")
    p.add_argument("--complete_missing", action="store_true",
                   help="For partially submitted runs, submit a complement batch for missing indices.")
    p.add_argument("--complete_failed", action="store_true",
                   help="For fully retrieved runs with error records, resubmit only the failed indices.")

    args = p.parse_args()

    datasets        = ALL_DATASETS if args.all_datasets else args.datasets
    output_dir      = Path(os.path.expanduser(args.output_dir))
    data_root       = os.path.expanduser(args.data_root)
    class_names_dir = os.path.expanduser(args.class_names_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = "__".join(filter(None, [
        "all_datasets" if args.all_datasets else "_".join(d.replace("-", "").replace("_", "") for d in datasets)[:40],
        args.name,
    ])) or "experiment"
    manifest_path = output_dir / f"batch_manifest__{tag}.json"

    if manifest_path.exists():
        manifest: list[dict] = json.loads(manifest_path.read_text())
        n = len(manifest)
        print(f"  (Found existing manifest with {n} entr{'y' if n == 1 else 'ies'})")
    else:
        manifest: list[dict] = []

    already_submitted: dict[str, dict] = {
        e["run_name"]: e for e in manifest if e.get("batch_id")
    }

    print()
    print("=" * 60)
    print(f"  Datasets:        {', '.join(datasets)}")
    print(f"  Models:          {len(MODELS)}")
    print(f"  Experiment:      {args.name or '<none>'}")
    print(f"  Complete missing: {args.complete_missing}")
    print(f"  Complete failed:  {args.complete_failed}")
    print(f"  Manifest:        {manifest_path}")
    print("=" * 60)
    print()

    new_submitted = 0
    new_failed    = 0
    skipped       = 0

    for dataset in datasets:
        dataset = resolve_dataset_name(dataset, data_root)
        print(f"── Dataset: {dataset}")

        for entry in MODELS:
            key      = entry["key"]
            provider = entry["provider"]
            model    = entry["model"]
            run_name = run_name_for(dataset, key, args.name)

            print(f"   {key:<18}  provider={provider}  model={model}")
            print(f"   run_name: {run_name}")

            predictions_path = output_dir / run_name / "predictions.jsonl"

            # --complete_failed: resubmit only error records from fully retrieved runs
            if args.complete_failed and predictions_path.exists():
                error_indices = []
                for line in predictions_path.read_text().splitlines():
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get("error", False):
                                error_indices.append(rec["index"])
                        except Exception:
                            pass
                if not error_indices:
                    print(f"   ✓ No errors — skipping")
                    print()
                    skipped += 1
                    continue
                print(f"   ⚠  {len(error_indices)} error(s) — resubmitting indices {error_indices}")
                complement_run_name = run_name + "__complement"
                batch_id = submit_job_simple(
                    provider        = provider,
                    model           = model,
                    dataset         = dataset,
                    run_name        = complement_run_name,
                    data_root       = data_root,
                    class_names_dir = class_names_dir,
                    output_dir      = str(output_dir),
                    split           = args.split,
                    max_samples     = None,
                    indices         = error_indices,
                )
                status = "submitted" if batch_id else "failed"
                print(f"   → complement batch_id: {batch_id}  [{status}]")
                print()
                manifest.append({
                    "dataset": dataset, "key": key, "provider": provider,
                    "model": model, "run_name": complement_run_name,
                    "batch_id": batch_id, "status": status, "experiment": args.name,
                })
                if status == "submitted":
                    new_submitted += 1
                else:
                    new_failed += 1
                continue

            # --complete_missing: skip if there is a pending/submitted (not yet retrieved) batch for this run
            has_active_batch = any(
                e.get("run_name") == run_name
                and e.get("batch_id")
                and e.get("status") not in ("retrieved", "failed", None)
                for e in manifest
            )

            # --complete_missing mode: only act on fully retrieved runs with no pending batch
            if args.complete_missing and predictions_path.exists() and not has_active_batch:
                done_indices  = get_done_indices(predictions_path)
                total_indices = get_total_indices(data_root, dataset, args.work_dir)
                missing       = sorted(set(total_indices) - done_indices)

                if not missing:
                    print(f"   ✓ Complete — {len(done_indices)}/{len(total_indices)} predictions present")
                    print()
                    skipped += 1
                    continue

                print(f"   ⚠  Missing {len(missing)}/{len(total_indices)} predictions — submitting complement batch")
                complement_run_name = run_name + "__complement"

                batch_id = submit_job_simple(
                    provider        = provider,
                    model           = model,
                    dataset         = dataset,
                    run_name        = complement_run_name,
                    data_root       = data_root,
                    class_names_dir = class_names_dir,
                    output_dir      = str(output_dir),
                    split           = args.split,
                    max_samples     = None,
                    indices         = missing,
                )

                # Note: llm_classify.py auto-resumes from existing predictions.jsonl
                # so pointing it at the same run_name with overwrite=False handles dedup.
                # Here we use a complement run then merge below.
                status = "submitted" if batch_id else "failed"
                print(f"   → complement batch_id: {batch_id}  [{status}]")
                print(f"   ℹ  After retrieving, merge with:")
                print(f"      python -c \"")
                print(f"import json; from pathlib import Path")
                print(f"p1 = Path('{predictions_path}')")
                print(f"p2 = Path('{output_dir / complement_run_name}/predictions.jsonl')")
                print(f"recs = [json.loads(l) for p in [p1,p2] for l in p.read_text().splitlines() if l.strip()]")
                print(f"seen = {{}}; deduped = [seen.setdefault(r['index'],r) for r in recs if r['index'] not in seen]")
                print(f"p1.write_text('\\n'.join(json.dumps(r) for r in sorted(deduped, key=lambda r: r['index']))+'\\n')")
                print(f"print(f'Merged {{len(deduped)}} records')\"")
                print()

                manifest.append({
                    "dataset": dataset, "key": key, "provider": provider,
                    "model": model, "run_name": complement_run_name,
                    "batch_id": batch_id, "status": status, "experiment": args.name,
                })
                if status == "submitted":
                    new_submitted += 1
                else:
                    new_failed += 1
                continue

            # In complete_missing mode, skip runs that have no predictions yet (pending/not submitted)
            if args.complete_missing and not predictions_path.exists():
                print(f"   ⏭  No predictions yet (pending or not submitted) — skipping in complete_missing mode")
                print()
                skipped += 1
                continue

            # Normal submission mode
            if run_name in already_submitted:
                existing = already_submitted[run_name]
                print(f"   ✓ Already submitted — batch_id: {existing['batch_id']}  [skipped]")
                print()
                skipped += 1
                continue

            batch_id = submit_job_simple(
                provider        = provider,
                model           = model,
                dataset         = dataset,
                run_name        = run_name,
                data_root       = data_root,
                class_names_dir = class_names_dir,
                output_dir      = str(output_dir),
                split           = args.split,
                max_samples     = args.max_samples,
            )

            status = "submitted" if batch_id else "failed"
            print(f"   → batch_id: {batch_id}  [{status}]")
            print()

            manifest.append({
                "dataset": dataset, "key": key, "provider": provider,
                "model": model, "run_name": run_name,
                "batch_id": batch_id, "status": status, "experiment": args.name,
            })
            if status == "submitted":
                new_submitted += 1
            else:
                new_failed += 1

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