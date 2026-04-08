#!/usr/bin/env python3
"""
run_openai_sequential.py — Submit OpenAI batch jobs one dataset at a time,
polling every 5 minutes until each job completes before moving to the next.
Handles Stanford Cars automatically by splitting into two 500-sample batches.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from openai import OpenAI

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "caltech101",
    # "fgvc-aircraft-2013b",
    "flowers-102",
    # "oxford-iiit-pet",
    # "stanford_cars",
    "uc-merced-land-use-dataset",
]

LARGE_DATASETS   = {"stanford_cars": 500}
DATASET_SIZES    = {"uc-merced-land-use-dataset": 420}
DEFAULT_DATASET_SIZE = 1000

POLL_INTERVAL = 60
PROVIDER      = "openai"
MODEL         = "gpt-4o-mini"
OUTPUT_DIR    = Path("./llm_classification_results")


# ---------------------------------------------------------------------------
# Archive verification
# ---------------------------------------------------------------------------

def verify_archive(archive_path: Path, expected: int) -> tuple[bool, int]:
    """Returns (ok, actual_count)."""
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                rows = list(csv.DictReader(
                    io.TextIOWrapper(tar.extractfile(tar.getmember(cand)))))
                return len(rows) == expected, len(rows)
            except KeyError:
                continue
    return False, 0


def find_and_verify_archive(dataset: str, data_root: str,
                             expected: int) -> Path | None:
    """
    Find the archive for this dataset under data_root, verify it has the
    correct number of images. Aborts if wrong.
    """
    root    = Path(os.path.expanduser(data_root))
    matches = sorted(root.glob(f"{dataset}*_processed.tar.zst"))
    if not matches:
        print(f"  ⚠ No archive found for {dataset!r} under {root}")
        return None
    archive = matches[0]
    ok, actual = verify_archive(archive, expected)
    if not ok:
        print(f"  ✗ Archive size mismatch: {archive.name}")
        print(f"    Expected {expected} images, found {actual}")
        print(f"    Fix with verify_and_repack.py before submitting.")
        return None
    print(f"  ✓ Archive verified: {archive.name} ({actual} images)")
    return archive


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def get_manifest_path(experiment: str) -> Path:
    return OUTPUT_DIR / f"batch_manifest__all_datasets__{experiment}.json"

def load_manifest(experiment: str) -> list[dict]:
    p = get_manifest_path(experiment)
    return json.loads(p.read_text()) if p.exists() else []

def save_manifest(manifest: list[dict], experiment: str):
    get_manifest_path(experiment).write_text(json.dumps(manifest, indent=2))

def upsert_manifest(manifest: list[dict], entry: dict) -> list[dict]:
    for i, e in enumerate(manifest):
        if e["run_name"] == entry["run_name"]:
            manifest[i] = entry
            return manifest
    manifest.append(entry)
    return manifest


# ---------------------------------------------------------------------------
# Batch submission / retrieval
# ---------------------------------------------------------------------------

def submit_batch(dataset: str, run_name: str, data_root: str,
                 class_names_dir: str, experiment: str,
                 max_samples: int | None = None,
                 indices: list[int] | None = None,
                 max_retries: int = 3) -> str | None:
    cmd = [
        sys.executable, "llm_classify.py",
        "--batch",
        "--provider",        PROVIDER,
        "--model",           MODEL,
        "--dataset",         dataset,
        "--split",           "test",
        "--data_root",       os.path.expanduser(data_root),
        "--class_names_dir", os.path.expanduser(class_names_dir),
        "--output_dir",      str(OUTPUT_DIR),
        "--run_name",        run_name,
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    if indices is not None:
        cmd += ["--indices", ",".join(map(str, indices))]

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait = 30 * attempt
            print(f"  ↺ Retry {attempt}/{max_retries} in {wait}s...")
            time.sleep(wait)

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ⚠  llm_classify.py failed (attempt {attempt}/{max_retries})")
            continue

        meta_path = OUTPUT_DIR / run_name / "batch_meta.json"
        if not meta_path.exists():
            print(f"  ⚠  batch_meta.json not found (attempt {attempt}/{max_retries})")
            continue

        meta = json.loads(meta_path.read_text())
        batch_id = meta.get("batch_id")
        if batch_id:
            return batch_id

    print(f"  ✗ Submission failed after {max_retries} attempts")
    return None


def retrieve_batch(run_name: str, dataset: str, data_root: str,
                   class_names_dir: str, batch_id: str) -> bool:
    cmd = [
        sys.executable, "llm_classify.py",
        "--batch_retrieve",  batch_id,
        "--provider",        PROVIDER,
        "--dataset",         dataset,
        "--split",           "test",
        "--data_root",       os.path.expanduser(data_root),
        "--class_names_dir", os.path.expanduser(class_names_dir),
        "--output_dir",      str(OUTPUT_DIR),
        "--run_name",        run_name,
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def poll_until_done(batch_id: str) -> str:
    client = OpenAI()
    while True:
        b         = client.batches.retrieve(batch_id)
        status    = b.status
        completed = b.request_counts.completed if b.request_counts else "?"
        total     = b.request_counts.total     if b.request_counts else "?"
        print(f"  [{time.strftime('%H:%M:%S')}] status={status}  {completed}/{total} completed")
        if status in ("completed", "failed", "expired", "cancelled"):
            return status
        print(f"  Sleeping {POLL_INTERVAL//60} min...")
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_done_indices(predictions_path: Path) -> set[int]:
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


def expected_size(dataset: str) -> int:
    return DATASET_SIZES.get(dataset, DEFAULT_DATASET_SIZE)


def predictions_complete(predictions_p: Path, dataset: str) -> bool:
    if not predictions_p.exists():
        return False
    return len(get_done_indices(predictions_p)) >= expected_size(dataset)


def merge_predictions(main_path: Path, complement_path: Path):
    recs = []
    for p in [main_path, complement_path]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    seen, deduped = {}, []
    for r in recs:
        if r["index"] not in seen:
            seen[r["index"]] = True
            deduped.append(r)
    deduped.sort(key=lambda r: r["index"])
    main_path.write_text("\n".join(json.dumps(r) for r in deduped) + "\n")
    print(f"  Merged {len(deduped)} records into {main_path.name}")


# ---------------------------------------------------------------------------
# Dataset name resolution
# ---------------------------------------------------------------------------

def resolve_dataset_name(dataset: str, data_root: str) -> str:
    root    = Path(os.path.expanduser(data_root))
    matches = sorted(root.glob(f"{dataset}*_processed.tar.zst"))
    if not matches:
        print(f"  ⚠  No archive found for {dataset!r} under {root} — using bare name")
        return dataset
    resolved = matches[0].name.replace("_processed.tar.zst", "")
    if resolved != dataset:
        print(f"  → Resolved {dataset!r}  →  {resolved!r}")
    return resolved


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def process_dataset(dataset: str, resolved_dataset: str,
                    args, manifest: list[dict]) -> list[dict]:

    experiment    = args.name
    max_samples   = LARGE_DATASETS.get(dataset)
    run_name      = f"{resolved_dataset}__{PROVIDER}__{experiment}"
    predictions_p = OUTPUT_DIR / run_name / "predictions.jsonl"
    exp_size      = expected_size(dataset)

    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset}  (resolved: {resolved_dataset})")
    print(f"  run_name : {run_name}")
    print(f"  expected : {exp_size} samples")
    print(f"{'='*60}")

    # ── Verify archive before doing anything ──────────────────
    archive = find_and_verify_archive(dataset, args.data_root, exp_size)
    if archive is None:
        print(f"  ✗ Skipping — archive missing or wrong size")
        return manifest

    # Already complete?
    if predictions_complete(predictions_p, dataset):
        print(f"  ✓ Already complete — skipping")
        return manifest

    # ── ADD THIS ──────────────────────────────────────────────
    existing = next((e for e in manifest if e["run_name"] == run_name), None)
    if existing and existing.get("status") == "retrieved":
        print(f"  ✓ Already retrieved in manifest — skipping")
        return manifest
    # ──────────────────────────────────────────────────────────

    # Phase 1
    existing = next((e for e in manifest if e["run_name"] == run_name), None)
    if existing and existing.get("batch_id") and existing.get("status") not in ("retrieved", "failed", None):
        print(f"  ↩  Resuming existing batch: {existing['batch_id']}")
        batch_id = existing["batch_id"]
    else:
        print(f"  → Submitting batch{'  (first 500)' if max_samples else ''}...")
        batch_id = submit_batch(
            dataset=resolved_dataset, run_name=run_name,
            data_root=args.data_root, class_names_dir=args.class_names_dir,
            experiment=experiment, max_samples=max_samples,
        )
        if not batch_id:
            print(f"  ✗ Submission failed — skipping dataset")
            return manifest
        print(f"  → batch_id: {batch_id}")
        manifest = upsert_manifest(manifest, {
            "dataset": resolved_dataset, "key": PROVIDER,
            "provider": PROVIDER, "model": MODEL,
            "run_name": run_name, "batch_id": batch_id,
            "status": "submitted", "experiment": experiment,
        })
        save_manifest(manifest, experiment)

    status = poll_until_done(batch_id)
    print(f"  → Batch finished: {status}")

    if status != "completed":
        manifest = upsert_manifest(manifest, {
            **next(e for e in manifest if e["run_name"] == run_name),
            "status": "failed"})
        save_manifest(manifest, experiment)
        return manifest

    retrieve_batch(run_name, resolved_dataset, args.data_root, args.class_names_dir, batch_id)
    manifest = upsert_manifest(manifest, {
        **next(e for e in manifest if e["run_name"] == run_name),
        "status": "retrieved"})
    save_manifest(manifest, experiment)

    # Phase 2: complement for large datasets
    if max_samples:
        done_indices = get_done_indices(predictions_p)
        missing      = sorted(set(range(exp_size)) - done_indices)
        if not missing:
            print(f"  ✓ All {exp_size} predictions present")
            return manifest

        print(f"\n  → Submitting complement batch ({len(missing)} missing indices)...")
        complement_run = run_name + "__complement"
        batch_id_2 = submit_batch(
            dataset=resolved_dataset, run_name=complement_run,
            data_root=args.data_root, class_names_dir=args.class_names_dir,
            experiment=experiment, indices=missing,
        )
        if not batch_id_2:
            print(f"  ✗ Complement submission failed")
            return manifest

        manifest = upsert_manifest(manifest, {
            "dataset": resolved_dataset, "key": PROVIDER,
            "provider": PROVIDER, "model": MODEL,
            "run_name": complement_run, "batch_id": batch_id_2,
            "status": "submitted", "experiment": experiment,
        })
        save_manifest(manifest, experiment)

        status2 = poll_until_done(batch_id_2)
        if status2 == "completed":
            retrieve_batch(complement_run, resolved_dataset,
                           args.data_root, args.class_names_dir, batch_id_2)
            merge_predictions(predictions_p,
                              OUTPUT_DIR / complement_run / "predictions.jsonl")
            manifest = upsert_manifest(manifest, {
                **next(e for e in manifest if e["run_name"] == complement_run),
                "status": "retrieved"})
            save_manifest(manifest, experiment)

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name",            required=True)
    p.add_argument("--data_root",       required=True)
    p.add_argument("--class_names_dir", default="~/data_processed/class_names")
    p.add_argument("--datasets",        nargs="+", default=ALL_DATASETS)
    p.add_argument("--poll_interval",   type=int, default=60)
    args = p.parse_args()

    global POLL_INTERVAL
    POLL_INTERVAL = args.poll_interval

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.name)

    print(f"\nExperiment : {args.name}")
    print(f"Data root  : {args.data_root}")
    print(f"Datasets   : {', '.join(args.datasets)}")
    print(f"Poll every : {POLL_INTERVAL//60} min")

    # Verify ALL archives first before submitting any jobs
    print(f"\nPre-flight archive check...")
    all_ok = True
    for dataset in args.datasets:
        archive = find_and_verify_archive(
            dataset, args.data_root, expected_size(dataset))
        if archive is None:
            all_ok = False

    if not all_ok:
        print(f"\n✗ Some archives are missing or wrong — fix with verify_and_repack.py first.")
        sys.exit(1)

    print(f"  All archives OK — proceeding.\n")

    for dataset in args.datasets:
        resolved = resolve_dataset_name(dataset, args.data_root)
        manifest = process_dataset(dataset, resolved, args, manifest)

    print("\n" + "="*60)
    print("  All datasets processed!")
    print("="*60)


if __name__ == "__main__":
    main()