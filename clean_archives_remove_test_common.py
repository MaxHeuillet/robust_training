#!/usr/bin/env python3
"""
clean_archives_remove_test_common.py

For each clean dataset archive in legolasflagstaff/RobustGenBench:
  1. Download the original .tar.zst
  2. Extract it
  3. Remove the test_common/ folder
  4. Repack as .tar.zst
  5. Verify the new archive (contents check)
  6. Upload back to the Hub

Includes a --dry-run mode that does everything except the upload, plus
an interactive confirmation before any destructive Hub operation.

Requirements:
    pip install huggingface_hub zstandard
    huggingface-cli login

Usage:
    # Dry run — process one dataset, verify, but don't upload
    python clean_archives_remove_test_common.py --dry-run --only uc-merced-land-use-dataset

    # Dry run — all datasets
    python clean_archives_remove_test_common.py --dry-run

    # Real run — will prompt before each upload
    python clean_archives_remove_test_common.py
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID   = "legolasflagstaff/RobustGenBench"
REPO_TYPE = "dataset"

DATASETS = [
    # "caltech101",
    # "fgvc-aircraft-2013b",
    # "flowers-102",
    # "oxford-iiit-pet",
    "stanford_cars",
    # "uc-merced-land-use-dataset",
]


def archive_filename(dataset: str) -> str:
    return f"{dataset}_processed.tar.zst"


def extract_zst_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as f, dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(path=dest_dir)


def repack_zst_tar(src_dir: Path, archive_path: Path, level: int = 3) -> None:
    cctx = zstd.ZstdCompressor(level=level)
    with open(archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as zst_stream:
            with tarfile.open(fileobj=zst_stream, mode="w|") as tar:
                # Walk in sorted order for determinism
                for root, dirs, files in os.walk(src_dir):
                    dirs.sort()
                    files.sort()
                    for file in files:
                        full = Path(root) / file
                        tar.add(full, arcname=str(full.relative_to(src_dir)))


def list_top_level_dirs(extracted_dir: Path) -> list[str]:
    return sorted(p.name for p in extracted_dir.iterdir() if p.is_dir())


def count_files_recursive(d: Path) -> int:
    return sum(1 for _ in d.rglob("*") if _.is_file())


def update_metadata(extracted_dir: Path) -> None:
    """If metadata.json exists, drop the test_common entry."""
    meta_path = extracted_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"    ⚠  Could not parse metadata.json ({e}); leaving it untouched")
        return

    splits = meta.get("splits", {})
    if "test_common" in splits:
        del splits["test_common"]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"    ✓ Removed test_common entry from metadata.json")


def process_dataset(dataset: str, work_root: Path, dry_run: bool, api: HfApi) -> None:
    print(f"\n{'='*70}")
    print(f"  {dataset}")
    print(f"{'='*70}")

    fname = archive_filename(dataset)
    work_dir   = work_root / dataset
    extract_dir = work_dir / "extracted"
    new_archive = work_dir / f"new_{fname}"

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download original
    print(f"  [1/6] Downloading {fname}...")
    downloaded = hf_hub_download(
        repo_id    = REPO_ID,
        repo_type  = REPO_TYPE,
        filename   = fname,
        local_dir  = str(work_dir),
    )
    downloaded = Path(downloaded)
    orig_size = downloaded.stat().st_size
    print(f"        downloaded: {downloaded}  ({orig_size/1e6:.1f} MB)")

    # 2. Extract
    print(f"  [2/6] Extracting...")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_zst_tar(downloaded, extract_dir)

    top_before = list_top_level_dirs(extract_dir)
    print(f"        contents before: {top_before}")

    # 3. Verify test_common is present
    tc_dir = extract_dir / "test_common"
    if not tc_dir.exists():
        print(f"  ⚠  test_common/ not found in archive — nothing to do, skipping")
        return

    tc_file_count = count_files_recursive(tc_dir)
    print(f"  [3/6] Removing test_common/ ({tc_file_count} files)...")
    shutil.rmtree(tc_dir)

    update_metadata(extract_dir)

    top_after = list_top_level_dirs(extract_dir)
    print(f"        contents after:  {top_after}")

    if "test_common" in top_after:
        print(f"  ✗ FAILED: test_common/ still present after removal — aborting this dataset")
        return

    # Sanity: make sure we still have test/
    if "test" not in top_after:
        print(f"  ✗ FAILED: test/ folder is missing after cleanup — aborting this dataset")
        return

    # 4. Repack
    print(f"  [4/6] Repacking...")
    repack_zst_tar(extract_dir, new_archive)
    new_size = new_archive.stat().st_size
    print(f"        new archive:  {new_archive.name}  ({new_size/1e6:.1f} MB)")
    print(f"        size delta:   {(new_size - orig_size)/1e6:+.1f} MB")

    # 5. Verify new archive by re-extracting to a temp dir and confirming structure
    print(f"  [5/6] Verifying new archive...")
    with tempfile.TemporaryDirectory(dir=str(work_dir)) as verify_dir:
        verify_path = Path(verify_dir)
        extract_zst_tar(new_archive, verify_path)
        verify_top = list_top_level_dirs(verify_path)
        print(f"        verified contents: {verify_top}")
        if "test_common" in verify_top:
            print(f"  ✗ VERIFICATION FAILED: test_common still appears in repacked archive")
            return
        if "test" not in verify_top:
            print(f"  ✗ VERIFICATION FAILED: test/ missing in repacked archive")
            return

    # 6. Upload (or skip in dry-run)
    if dry_run:
        print(f"  [6/6] DRY RUN — not uploading. New archive kept at:")
        print(f"        {new_archive}")
        return

    print(f"\n  About to upload {new_archive.name} to {REPO_ID}, replacing the existing file.")
    answer = input(f"  Proceed with upload? Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print(f"  Skipped. New archive remains at: {new_archive}")
        return

    print(f"  [6/6] Uploading...")
    api.upload_file(
        path_or_fileobj   = str(new_archive),
        path_in_repo      = fname,
        repo_id           = REPO_ID,
        repo_type         = REPO_TYPE,
        commit_message    = f"Remove unused test_common/ from {fname}",
    )
    print(f"  ✓ Uploaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except upload. New archives are kept on disk for inspection.")
    parser.add_argument("--only", nargs="+", choices=DATASETS, default=None,
                        help="Process only these datasets (default: all)")
    parser.add_argument("--work-dir", default=str(Path.home() / "robustgenbench_cleanup"),
                        help="Where to download and repack (default: ~/robustgenbench_cleanup)")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Don't delete the work directory at the end")
    args = parser.parse_args()

    work_root = Path(args.work_dir).expanduser()
    work_root.mkdir(parents=True, exist_ok=True)

    targets = args.only if args.only else DATASETS

    print(f"Repo:        {REPO_ID}")
    print(f"Datasets:    {', '.join(targets)}")
    print(f"Work dir:    {work_root}")
    print(f"Dry run:     {args.dry_run}")
    if not args.dry_run:
        print(f"\n⚠  This will REPLACE the archives on the Hub. You will be prompted before each upload.")
        confirm = input("Continue? Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    api = HfApi()

    for dataset in targets:
        try:
            process_dataset(dataset, work_root, args.dry_run, api)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ ERROR processing {dataset}: {type(e).__name__}: {e}")
            print(f"  Continuing with next dataset...")

    if not args.keep_work_dir and not args.dry_run:
        print(f"\nCleaning up work directory: {work_root}")
        shutil.rmtree(work_root, ignore_errors=True)
    else:
        print(f"\nWork directory preserved at: {work_root}")

    print("\nDone.")


if __name__ == "__main__":
    main()