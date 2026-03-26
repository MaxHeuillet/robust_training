#!/usr/bin/env python3
"""
analyze_perturbations.py — Compare each adversarial archive against the
clean archive, matched by position in labels.csv.

Both archives use the same 1000 test filenames (00000.png ... 00999.png)
under test/. The clean archive also stores train/val images under test/ with
overlapping filenames — fixed by filtering to test/ prefix only when loading.

Output: summary table + JSON report.
"""

import csv, io, json, os, sys, tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(os.path.expanduser("~/data"))
ADV_ROOT    = DATA_ROOT / "adversarial"
CLEAN_ROOT  = Path("/tmp/robustgenbench/data_processed")
OUTPUT_JSON = Path(os.path.expanduser("~/Desktop/perturbation_analysis.json"))

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

ADV_SUBPATHS = [
    ("L∞ ε=8/255",  "zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"),
    ("L∞ ε=30/255", "zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
    ("L2 ε=2",      "zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard"),
    ("L2 ε=8",      "zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard"),
    ("L1 ε=75",     "zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard"),
    ("L1 ε=300",    "zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard"),
    ("Common s=3",  "common/common_severity3"),
]

# ---------------------------------------------------------------------------
# Archive loading — test/ prefix filter prevents train/val collisions
# ---------------------------------------------------------------------------

def load_archive(archive_path: Path) -> list[bytes]:
    """
    Returns images in labels.csv order.
    Only loads images under test/ to avoid filename collisions with
    train/val images in the clean archive.
    """
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        raw_by_name = {}
        for member in tar.getmembers():
            if member.name.startswith("test/") and (
                    member.name.endswith(".png") or member.name.endswith(".jpg")):
                raw_by_name[Path(member.name).name] = tar.extractfile(member).read()
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f    = tar.extractfile(tar.getmember(cand))
                rows = list(csv.DictReader(io.TextIOWrapper(f)))
                break
            except KeyError:
                continue
    return [raw_by_name.get(r["filename"]) for r in rows]


def find_archive(dataset: str, subpath: str) -> Path | None:
    folder  = ADV_ROOT / subpath
    matches = sorted(folder.glob(f"{dataset}*_processed.tar.zst")) if folder.exists() else []
    return matches[0] if matches else None


def to_array(raw: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

results = {}

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    clean_archive = CLEAN_ROOT / f"{dataset}_processed.tar.zst"
    if not clean_archive.exists():
        print(f"  ⚠ Clean archive not found — skipping"); continue

    print(f"  Loading clean archive...")
    clean_imgs = load_archive(clean_archive)
    print(f"  Clean images loaded: {len(clean_imgs)}")

    results[dataset] = {}

    for label, subpath in ADV_SUBPATHS:
        archive = find_archive(dataset, subpath)
        if archive is None:
            print(f"  {label:<14} — not found")
            results[dataset][label] = None
            continue

        adv_imgs = load_archive(archive)
        n        = min(len(clean_imgs), len(adv_imgs))

        n_identical = 0
        max_diffs   = []
        mean_diffs  = []

        for i in tqdm(range(n), desc=f"  {label:<14}", leave=False):
            rc, ra = clean_imgs[i], adv_imgs[i]
            if rc is None or ra is None:
                continue
            a    = to_array(rc)
            b    = to_array(ra)
            diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
            max_d = float(diff.max())
            max_diffs.append(max_d)
            mean_diffs.append(float(diff.mean()))
            if max_d <= 1:   # tolerance for PNG round-trip noise
                n_identical += 1

        pct  = round(n_identical / n * 100, 2)
        flag = "⚠ HIGH" if pct > 5 else "✓"
        print(f"  {label:<14} n={n}  identical={n_identical} ({pct}%)  "
              f"Δmax_avg={round(float(np.mean(max_diffs)),2)}  {flag}")

        results[dataset][label] = {
            "n":             n,
            "n_identical":   n_identical,
            "pct_identical": pct,
            "mean_max_diff": round(float(np.mean(max_diffs)), 2),
            "mean_mean_diff": round(float(np.mean(mean_diffs)), 4),
        }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

labels = [l for l, _ in ADV_SUBPATHS]
col_w  = 14

print(f"\n\n{'='*90}")
print(f"  SUMMARY — % identical to clean (0% = all attacked, >0% = surrogate was wrong)")
print(f"{'='*90}")
print(f"{'Dataset':<32}" + "".join(f"{l:>{col_w}}" for l in labels))
print("─" * (32 + col_w * len(labels)))

for dataset in DATASETS:
    if dataset not in results:
        continue
    row = f"{dataset:<32}"
    for label in labels:
        s = results[dataset].get(label)
        if s is None:
            row += f"{'N/A':>{col_w}}"
        else:
            pct = s["pct_identical"]
            val = f"{'⚠' if pct > 5 else ''}{pct:.1f}%"
            row += f"{val:>{col_w}}"
    print(row)

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\n✓ Saved → {OUTPUT_JSON}")