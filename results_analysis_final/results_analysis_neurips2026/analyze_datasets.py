"""
download_and_analyze.py
=======================
1. Downloads the _processed.tar.zst archives from MaxHeuillet/RobustGenBench
   on HuggingFace into a destination folder (default: ~/data/processed).
2. Extracts them in-place (skips already-extracted archives).
3. Reads metadata.json + split/labels.csv from each archive to report:
     - train / test / val split sizes
     - number of classes
     - class distribution (with names if ~/data/class_names/<dataset>.txt exists)
     - whether the task is coarse or fine-grained

Archive contents:
    metadata.json                  -> N (n_classes), splits.{split}.count
    {train|val|test|test_common}/
        labels.csv                 -> filename,label  (integer class ids)
        <images>.png

Requirements:
    pip install huggingface_hub zstandard

Usage:
    python download_and_analyze.py                          # download + analyse
    python download_and_analyze.py --analyze-only           # skip download
    python download_and_analyze.py --dest ~/data/processed
    python download_and_analyze.py --class-names ~/data/class_names
"""

import argparse
import csv
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import zstandard as zstd
except ImportError:
    sys.exit("ERROR: 'zstandard' not found.\nInstall: pip install zstandard")
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    sys.exit("ERROR: 'huggingface_hub' not found.\nInstall: pip install huggingface_hub")

# ── config ────────────────────────────────────────────────────────────────────

HF_REPO   = "MaxHeuillet/RobustGenBench"
REPO_TYPE = "dataset"

FINE_GRAINED   = {"fgvc-aircraft", "flowers-102", "stanford_cars", "oxford-iiit-pet"}
COARSE_GRAINED = {"caltech101", "uc-merced"}

def granularity(name: str) -> str:
    key = name.lower()
    for fg in FINE_GRAINED:
        if fg in key:
            return "fine-grained"
    for cg in COARSE_GRAINED:
        if cg in key:
            return "coarse-grained"
    return "unknown"


# ── class name loading ────────────────────────────────────────────────────────

def load_class_names(dataset_name: str, class_names_dir: Path) -> dict:
    """
    Try to load class names from <class_names_dir>/<dataset_name>.txt
    Expected format: one class name per line, index = line number (0-based).
    Returns {int_id: class_name} or empty dict if not found.
    """
    if not class_names_dir.exists():
        return {}
    # Try exact match then fuzzy
    candidates = [
        class_names_dir / f"{dataset_name}.txt",
        *[p for p in class_names_dir.iterdir()
          if p.suffix == ".txt" and dataset_name.split("_")[0].lower() in p.stem.lower()],
    ]
    for path in candidates:
        if path.exists():
            names = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            return {i: name for i, name in enumerate(names)}
    return {}


# ── download helpers ──────────────────────────────────────────────────────────

def list_processed_archives(token=None) -> list:
    files = list_repo_files(HF_REPO, repo_type=REPO_TYPE, token=token)
    return sorted(f for f in files if f.endswith("_processed.tar.zst") and "/" not in f)


def download_archive(filename: str, dest_dir: Path, token=None) -> Path:
    print(f"  ↓  {filename}")
    return Path(hf_hub_download(
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(dest_dir),
        token=token,
    ))


def extract_zst_tar(archive_path: Path, extract_to: Path) -> None:
    print(f"  ✦  extracting {archive_path.name} …")
    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                try:
                    tf.extractall(path=str(extract_to), filter="data")
                except TypeError:
                    tf.extractall(path=str(extract_to))  # Python < 3.12 fallback
    print(f"     → done")


# ── analysis ──────────────────────────────────────────────────────────────────

def read_labels_csv(tf: tarfile.TarFile, path: str) -> list[int]:
    """Extract and parse a labels.csv member from an open TarFile."""
    try:
        member = tf.getmember(path)
    except KeyError:
        return []
    content = tf.extractfile(member).read().decode()
    reader = csv.DictReader(io.StringIO(content))
    return [int(row["label"]) for row in reader]


def analyze_archive(archive_path: Path, class_names: dict) -> dict:
    """
    Stream through the archive once, collecting metadata + all label lists.
    Never fully decompresses to disk.
    """
    meta = {}
    labels_by_split: dict[str, list[int]] = {}

    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for member in tf:
                    if member.name == "metadata.json":
                        meta = json.loads(tf.extractfile(member).read())
                    elif member.name.endswith("labels.csv"):
                        split = member.name.split("/")[0]   # e.g. "train"
                        content = tf.extractfile(member).read().decode()
                        reader_csv = csv.DictReader(io.StringIO(content))
                        labels_by_split[split] = [int(row["label"]) for row in reader_csv]

    n_classes = meta.get("N", 0)
    splits_meta = meta.get("splits", {})

    # Build per-split counts and overall class distribution
    result = {"n_classes": n_classes, "splits": {}, "distribution": Counter()}

    for split, labels in labels_by_split.items():
        result["splits"][split] = len(labels)
        if split == "test":
            result["distribution"].update(labels)

    # Fallback to metadata counts if CSVs were missing
    for split, info in splits_meta.items():
        if split not in result["splits"]:
            result["splits"][split] = info.get("count", 0)

    # Replace integer keys with class names if available
    if class_names:
        result["distribution"] = Counter(
            {class_names.get(k, f"class_{k}"): v
             for k, v in result["distribution"].items()}
        )

    return result


# ── pretty print ──────────────────────────────────────────────────────────────

def fmt_dist(dist: Counter, top_n: int = 10) -> str:
    if not dist:
        return "  (label ids only — no class_names file found)"
    total = sum(dist.values())
    lines = []
    for cls, cnt in dist.most_common(top_n):
        pct = 100 * cnt / total if total else 0
        lines.append(f"  {str(cls):<48s} {cnt:>6,d}  ({pct:5.1f}%)")
    if len(dist) > top_n:
        lines.append(f"  … and {len(dist) - top_n} more classes")
    return "\n".join(lines)


def print_result(name: str, info: dict) -> None:
    sp = info["splits"]
    train = sp.get("train", 0) + sp.get("val", 0)
    test  = sp.get("test", 0)
    val   = sp.get("val", 0)
    test_common = sp.get("test_common", 0)

    print("=" * 70)
    print(f"Dataset      : {name}")
    print(f"Task         : {granularity(name)}")
    print(f"Classes      : {info['n_classes']}")
    print(f"Train        : {train:,}  (train={sp.get('train',0):,} + val={val:,})")
    print(f"Test         : {test:,}")
    if test_common:
        print(f"Test (common): {test_common:,}")
    print(f"Class distribution — test (top 10):")
    print(fmt_dist(info["distribution"]))

    # Missing classes (present in train+val but absent from test)
    n_classes = info["n_classes"]
    represented = set(info["distribution"].keys())
    # represented keys may be class names (str) or int ids — normalise to compare
    if n_classes > 0:
        all_ids = set(range(n_classes))
        # if distribution was resolved to names, we can only check by count
        if all(isinstance(k, int) for k in represented):
            missing = sorted(all_ids - represented)
        else:
            missing = []
            if len(represented) < n_classes:
                missing = [f"({n_classes - len(represented)} unnamed classes missing)"]
        if missing:
            print(f"  ⚠  Missing from test ({len(missing)} classes): {missing[:20]}"
                  + (" …" if len(missing) > 20 else ""))
        else:
            print(f"  ✔  All {n_classes} classes represented in test split")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dest",         default="~/data/processed",   help="Download/extract destination")
    parser.add_argument("--class-names",  default="~/data/class_names", help="Dir with <dataset>.txt class name files")
    parser.add_argument("--analyze-only", action="store_true",           help="Skip download, analyse existing archives")
    parser.add_argument("--token",        default=None,                  help="HuggingFace token (private repos)")
    args = parser.parse_args()

    dest             = Path(args.dest).expanduser()
    class_names_dir  = Path(args.class_names).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    # ── download ──────────────────────────────────────────────────────────────
    if not args.analyze_only:
        print(f"\n── Listing archives on {HF_REPO} ──")
        archives = list_processed_archives(token=args.token)
        if not archives:
            sys.exit("No *_processed.tar.zst files found. Check repo name / token.")
        print(f"Found {len(archives)} archive(s)\n")
        for fname in archives:
            archive_path = dest / fname
            if archive_path.exists():
                print(f"  ✔  {fname} already downloaded — skipping")
                continue
            download_archive(fname, dest, token=args.token)

    # ── analyse ───────────────────────────────────────────────────────────────
    print("\n── Analysis ──\n")
    archives = sorted(dest.glob("*_processed.tar.zst"))
    if not archives:
        sys.exit(f"No *_processed.tar.zst archives found in {dest}.")

    all_results = []
    for archive_path in archives:
        name = archive_path.name.replace("_processed.tar.zst", "")
        print(f"Reading {archive_path.name} …")
        class_names = load_class_names(name, class_names_dir)
        info = analyze_archive(archive_path, class_names)
        print_result(name, info)

        sp = info["splits"]
        all_results.append({
            "name":               name,
            "granularity":        granularity(name),
            "n_classes":          info["n_classes"],
            "train_size":         sp.get("train", 0) + sp.get("val", 0),
            "test_size":          sp.get("test", 0),
            "test_common_size":   sp.get("test_common", 0),
            "class_distribution": dict(info["distribution"].most_common()),
        })

    out = dest / "dataset_summary.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON summary saved → {out}")


if __name__ == "__main__":
    main()