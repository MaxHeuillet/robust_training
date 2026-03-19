#!/usr/bin/env python3
"""
retrieve_batches.py — Retrieve results for batch jobs recorded in a manifest.

Safe to re-run at any time:
  - Already-retrieved jobs (predictions.jsonl present) are skipped.
  - Still-pending jobs are reported and left for the next pass.
  - The manifest is updated in-place with the latest status.

Usage:
    python retrieve_batches.py <manifest.json>

    # Override paths if needed
    python retrieve_batches.py <manifest.json> \
        --data_root ~/data_processed \
        --class_names_dir ~/data_processed/class_names \
        --output_dir ./results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Keywords in llm_classify.py output that indicate a job isn't done yet
PENDING_SIGNALS = [
    "not yet complete",
    "processing",
    "JOB_STATE_RUNNING",
    "JOB_STATE_PENDING",
    "in_progress",
    "validating",
]

# Keywords that indicate a job has permanently failed and should be retried
FAILED_SIGNALS = [
    "status: failed",
    "job did not succeed",
    "JOB_STATE_FAILED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_CANCELLED",
]

# ---------------------------------------------------------------------------

def merge_complement(run_name: str, output_dir: str):
    """If a __complement predictions file exists, merge it into the main run."""
    base = Path(output_dir) / run_name
    comp = Path(output_dir) / (run_name + "__complement")
    p1   = base / "predictions.jsonl"
    p2   = comp / "predictions.jsonl"

    if not p2.exists():
        return

    recs = [json.loads(l) for p in [p1, p2] for l in p.read_text().splitlines() if l.strip()]
    seen = {}
    for r in recs:
        if r["index"] not in seen:
            seen[r["index"]] = r
    deduped = sorted(seen.values(), key=lambda r: r["index"])
    with open(p1, "w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")
    print(f"   ✓ Merged complement — {len(deduped)} total predictions")


def score_predictions(predictions_path: Path) -> dict:
    records = [json.loads(l) for l in predictions_path.read_text().splitlines() if l.strip()]
    total   = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    errors  = sum(1 for r in records if r.get("error"))
    valid   = total - errors
    acc     = correct / valid if valid > 0 else 0.0
    return {"total": total, "correct": correct, "errors": errors, "valid": valid, "accuracy": acc}


def retrieve_job(entry: dict, data_root: str, class_names_dir: str, output_dir: str) -> str:
    """
    Call llm_classify.py --batch_retrieve.
    Returns one of: "retrieved", "pending", "failed"
    """
    predictions_path = Path(output_dir) / entry["run_name"] / "predictions.jsonl"

    # Already done
    if predictions_path.exists():
        return "retrieved"

    cmd = [
        sys.executable, "llm_classify.py",
        "--batch_retrieve",  entry["batch_id"],
        "--provider",        entry["provider"],
        "--dataset",         entry["dataset"],
        "--data_root",       data_root,
        "--class_names_dir", class_names_dir,
        "--output_dir",      output_dir,
        "--run_name",        entry["run_name"],
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, end="")

    output_lower = output.lower()

    # Permanent failures — mark as failed so launcher will retry
    if any(sig.lower() in output_lower for sig in FAILED_SIGNALS):
        return "failed"

    # Still running
    if any(sig.lower() in output_lower for sig in PENDING_SIGNALS):
        return "pending"

    # Check if predictions file was created
    if predictions_path.exists():
        return "retrieved"

    return "failed"


def main():
    p = argparse.ArgumentParser(description="Retrieve batch classification results")

    p.add_argument("manifest", help="Path to manifest JSON produced by launch_batches.py")

    # Path overrides (fall back to values stored in manifest entries if not given)
    p.add_argument("--data_root",       default=None)
    p.add_argument("--class_names_dir", default=None)
    p.add_argument("--output_dir",      default=None)

    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    manifest: list[dict] = json.loads(manifest_path.read_text())

    # Defaults — read from first entry's run_name path or fall back to cwd defaults
    data_root       = os.path.expanduser(args.data_root       or "~/data_processed")
    class_names_dir = os.path.expanduser(args.class_names_dir or "~/data_processed/class_names")
    output_dir      = os.path.expanduser(args.output_dir      or "./llm_classification_results")

    print()
    print("=" * 60)
    print(f"  Manifest:  {manifest_path}")
    print(f"  Jobs:      {len(manifest)}")
    print("=" * 60)
    print()

    counts = {"retrieved": 0, "pending": 0, "failed": 0, "skipped_no_id": 0}

    for entry in manifest:
        key      = entry["key"]
        dataset  = entry["dataset"]
        batch_id = entry.get("batch_id")
        run_name = entry["run_name"]

        print(f"── {dataset} / {key}")
        print(f"   run_name : {run_name}")
        print(f"   batch_id : {batch_id}")

        if not batch_id:
            print("   ⚠  No batch_id — was submission successful?")
            entry["status"] = "failed"
            counts["skipped_no_id"] += 1
            print()
            continue

        # Already retrieved in a prior pass
        predictions_path = Path(output_dir) / run_name / "predictions.jsonl"
        if entry.get("status") == "retrieved" and predictions_path.exists():
            merge_complement(run_name, output_dir)
            scores = score_predictions(predictions_path)
            print(f"   ✓ Already retrieved — acc={scores['accuracy']:.4f} "
                  f"({scores['correct']}/{scores['valid']}, {scores['errors']} errors)")
            counts["retrieved"] += 1
            print()
            continue

        print("   → Attempting retrieval...")
        status = retrieve_job(entry, data_root, class_names_dir, output_dir)
        entry["status"] = status

        if status == "retrieved":
            if run_name.endswith("__complement"):
                parent_run = run_name[:-len("__complement")]
                merge_complement(parent_run, output_dir)
                parent_path = Path(output_dir) / parent_run / "predictions.jsonl"
                scores = score_predictions(parent_path if parent_path.exists() else predictions_path)
            else:
                merge_complement(run_name, output_dir)
                scores = score_predictions(predictions_path)
            print(f"   ✓ Retrieved — acc={scores['accuracy']:.4f} "
                  f"({scores['correct']}/{scores['valid']}, {scores['errors']} errors)")
            counts["retrieved"] += 1
        elif status == "pending":
            print("   ⏳ Still pending")
            counts["pending"] += 1
        else:
            # Reset batch_id so the launcher will resubmit this job
            print("   ✗ Permanently failed — resetting for resubmission")
            entry["batch_id"] = None
            entry["status"]   = "failed"
            counts["failed"] += 1

        print()

    # Save updated manifest
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ✓ Retrieved : {counts['retrieved']}")
    print(f"  ⏳ Pending   : {counts['pending']}")
    print(f"  ✗ Failed    : {counts['failed']}")
    print()

    # Accuracy table for retrieved jobs
    retrieved_entries = [e for e in manifest if e.get("status") == "retrieved" and not e["run_name"].endswith("__complement")]
    if retrieved_entries:
        print(f"  {'Dataset':<30} {'Model key':<20} {'Accuracy':>9}  {'Correct':>8}  {'Errors':>7}")
        print(f"  {'-'*80}")
        for entry in retrieved_entries:
            predictions_path = Path(output_dir) / entry["run_name"] / "predictions.jsonl"
            if predictions_path.exists():
                s = score_predictions(predictions_path)
                print(f"  {entry['dataset']:<30} {entry['key']:<20} "
                      f"{s['accuracy']:>9.4f}  {s['correct']:>5}/{s['valid']:<5}  {s['errors']:>7}")
        print()

    if counts["pending"] > 0:
        print(f"  {counts['pending']} job(s) still pending. Re-run when ready:")
        print(f"    python retrieve_batches.py {manifest_path}")
        print()

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()