"""
download_and_analyze.py
=======================
1. Downloads the _processed.tar.zst archives from MaxHeuillet/RobustGenBench
   on HuggingFace into ~/data/processed/.
2. Extracts them in-place.
3. Analyses each dataset and reports:
     - train / test split sizes
     - number of classes
     - class distribution
     - whether the task is coarse or fine-grained

Requirements (install once in your env):
    pip install huggingface_hub zstandard

Usage:
    python download_and_analyze.py                    # downloads + analyses
    python download_and_analyze.py --analyze-only     # skip download, just analyse
    python download_and_analyze.py --dest ~/data/processed
"""

import argparse
import io
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import zstandard as zstd
except ImportError:
    sys.exit(
        "ERROR: 'zstandard' not found.\n"
        "Install it with:  pip install zstandard"
    )
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    sys.exit(
        "ERROR: 'huggingface_hub' not found.\n"
        "Install it with:  pip install huggingface_hub"
    )

# ── config ────────────────────────────────────────────────────────────────────

HF_REPO      = "MaxHeuillet/RobustGenBench"
REPO_TYPE    = "dataset"
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm", ".pgm"}

FINE_GRAINED   = {"fgvc-aircraft", "flowers-102", "stanford_cars", "oxford-iiit-pet"}
COARSE_GRAINED = {"caltech101", "ucmerced_landuse", "uc-merced"}

def granularity(name: str) -> str:
    key = name.lower()
    for fg in FINE_GRAINED:
        if fg in key:
            return "fine-grained"
    for cg in COARSE_GRAINED:
        if cg in key:
            return "coarse-grained"
    return "unknown"


# ── download helpers ──────────────────────────────────────────────────────────

def list_processed_archives(token: str | None = None) -> list[str]:
    """Return the list of *_processed.tar.zst filenames in the repo root."""
    files = list_repo_files(HF_REPO, repo_type=REPO_TYPE, token=token)
    return [f for f in files if f.endswith("_processed.tar.zst") and "/" not in f]


def download_archive(filename: str, dest_dir: Path, token: str | None = None) -> Path:
    """Download one archive via huggingface_hub (uses local cache automatically)."""
    print(f"  ↓  {filename}")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(dest_dir),
        token=token,
    )
    return Path(local_path)


def extract_zst_tar(archive_path: Path, extract_to: Path) -> None:
    """Extract a .tar.zst archive using streaming decompression."""
    print(f"  ✦  extracting {archive_path.name} …")
    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                tf.extractall(path=str(extract_to))
    print(f"     → extracted to {extract_to}")


# ── analysis helpers ──────────────────────────────────────────────────────────

def count_class_folders(directory: Path) -> Counter:
    dist: Counter = Counter()
    if not directory.exists():
        return dist
    for cls_dir in sorted(directory.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
            if n:
                dist[cls_dir.name] = n
    return dist


def analyze_dataset(ds_root: Path) -> dict:
    """
    Expected layout produced by your processing script:
        <dataset_name>/
            train/<class>/<images>
            test/<class>/<images>

    Falls back gracefully if only one split exists.
    """
    train_dir = ds_root / "train"
    test_dir  = ds_root / "test"
    val_dir   = ds_root / "val"

    train_dist = count_class_folders(train_dir) if train_dir.exists() else Counter()
    test_dist  = count_class_folders(test_dir)  if test_dir.exists()  else Counter()

    # merge val into train if no dedicated test
    if not test_dist and val_dir.exists():
        test_dist = count_class_folders(val_dir)

    all_dist   = train_dist + test_dist
    n_classes  = len(all_dist) or len(count_class_folders(ds_root))

    return {
        "train":      sum(train_dist.values()),
        "test":       sum(test_dist.values()),
        "n_classes":  n_classes,
        "distribution": all_dist,
    }


# ── pretty print ──────────────────────────────────────────────────────────────

def fmt_dist(dist: Counter, top_n: int = 10) -> str:
    if not dist:
        return "  (no distribution data)"
    total = sum(dist.values())
    lines = []
    for cls, cnt in dist.most_common(top_n):
        pct = 100 * cnt / total if total else 0
        lines.append(f"  {cls:<48s} {cnt:>6,d}  ({pct:5.1f}%)")
    if len(dist) > top_n:
        lines.append(f"  … and {len(dist) - top_n} more classes")
    return "\n".join(lines)


def print_result(name: str, info: dict) -> None:
    grain = granularity(name)
    print("=" * 70)
    print(f"Dataset    : {name}")
    print(f"Task       : {grain}")
    print(f"Classes    : {info['n_classes']}")
    print(f"Train      : {info['train']:,} images")
    print(f"Test       : {info['test']:,} images")
    print("Class distribution (top 10):")
    print(fmt_dist(info["distribution"]))
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest",         default="~/data/processed", help="Where to download & extract archives")
    parser.add_argument("--analyze-only", action="store_true",        help="Skip download; analyse existing extracts")
    parser.add_argument("--token",        default=None,               help="HuggingFace token (if repo is private)")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    # ── download + extract ────────────────────────────────────────────────────
    if not args.analyze_only:
        print(f"\n── Listing archives on {HF_REPO} ──")
        archives = list_processed_archives(token=args.token)
        if not archives:
            print("No *_processed.tar.zst files found in repo root. Check repo name / token.")
            sys.exit(1)
        print(f"Found {len(archives)} archive(s): {', '.join(archives)}\n")

        for fname in sorted(archives):
            archive_path = dest / fname
            ds_name      = fname.replace("_processed.tar.zst", "")
            extract_dir  = dest / ds_name

            if extract_dir.exists() and any(extract_dir.iterdir()):
                print(f"  ✔  {ds_name} already extracted — skipping download")
                continue

            downloaded = download_archive(fname, dest, token=args.token)
            extract_zst_tar(downloaded, dest)

            # Optional: remove the archive after extraction to save disk space
            # downloaded.unlink()

    # ── analyse ───────────────────────────────────────────────────────────────
    print("\n── Analysis ──\n")
    dataset_dirs = sorted(
        p for p in dest.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not dataset_dirs:
        print(f"No extracted dataset folders found in {dest}.")
        sys.exit(1)

    all_results = []
    for ds_path in dataset_dirs:
        info = analyze_dataset(ds_path)
        print_result(ds_path.name, info)
        all_results.append({
            "name":               ds_path.name,
            "granularity":        granularity(ds_path.name),
            "n_classes":          info["n_classes"],
            "train_size":         info["train"],
            "test_size":          info["test"],
            "class_distribution": dict(info["distribution"].most_common()),
        })

    out = dest / "dataset_summary.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON summary saved → {out}")


if __name__ == "__main__":
    main()