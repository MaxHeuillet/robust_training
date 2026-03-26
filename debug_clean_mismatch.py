#!/usr/bin/env python3
"""
debug_clean_mismatch.py — Investigate the ordering mismatch between the
local clean archive and the adversarial archives.

Checks:
1. Do the labels.csv files have the same (filename, label) pairs?
2. Are they in the same order?
3. If not: what IS the mapping? Can we recover the correct correspondence?
4. How many images from the adversarial archive can be found in the clean
   archive by pixel content (brute-force on a small sample)?
"""

import csv, io, os, tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import zstandard as zstd

CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")
ADV_ROOT   = Path(os.path.expanduser(
    "~/data/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"))

DATASET = "caltech101"

# ---------------------------------------------------------------------------

def read_full_archive(archive_path: Path) -> tuple[list[dict], dict[str, bytes]]:
    """Returns (csv_rows, {filename: raw_bytes})"""
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    rows, raw_by_name = [], {}
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.endswith(".png") or name.endswith(".jpg"):
                raw_by_name[Path(name).name] = tar.extractfile(member).read()
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f = tar.extractfile(tar.getmember(cand))
                rows = list(csv.DictReader(io.TextIOWrapper(f)))
                break
            except KeyError:
                continue
    return rows, raw_by_name


def to_arr(raw: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


print(f"Loading clean archive for {DATASET}...")
clean_archive = CLEAN_ROOT / f"{DATASET}_processed.tar.zst"
clean_rows, clean_imgs = read_full_archive(clean_archive)

print(f"Loading adversarial archive...")
adv_archive = sorted(ADV_ROOT.glob(f"{DATASET}*_processed.tar.zst"))[0]
adv_rows, adv_imgs = read_full_archive(adv_archive)

print(f"\nClean  : {len(clean_rows)} rows, {len(clean_imgs)} images")
print(f"Adv    : {len(adv_rows)} rows, {len(adv_imgs)} images")

# ---------------------------------------------------------------------------
# 1. Check label set overlap
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("1. Label/filename overlap")
print(f"{'='*60}")

clean_pairs = {(r["filename"], r["label"]) for r in clean_rows}
adv_pairs   = {(r["filename"], r["label"]) for r in adv_rows}

print(f"Exact (filename, label) pairs in common: {len(clean_pairs & adv_pairs)}")
print(f"Only in clean : {len(clean_pairs - adv_pairs)}")
print(f"Only in adv   : {len(adv_pairs - clean_pairs)}")

# Check filenames only
clean_files = {r["filename"]: r["label"] for r in clean_rows}
adv_files   = {r["filename"]: r["label"] for r in adv_rows}
shared_fnames = set(clean_files.keys()) & set(adv_files.keys())
print(f"\nShared filenames: {len(shared_fnames)}")
label_match = sum(1 for f in shared_fnames if clean_files[f] == adv_files[f])
print(f"Of those, same label: {label_match}")

# ---------------------------------------------------------------------------
# 2. Check positional ordering
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("2. Positional ordering check (first 10 rows)")
print(f"{'='*60}")
print(f"{'idx':<6} {'clean_file':<14} {'clean_lbl':<12} {'adv_file':<14} {'adv_lbl':<10} {'same?'}")
for i in range(10):
    cf, cl = clean_rows[i]["filename"], clean_rows[i]["label"]
    af, al = adv_rows[i]["filename"],   adv_rows[i]["label"]
    same = "✓" if cf == af and cl == al else "✗"
    print(f"{i:<6} {cf:<14} {cl:<12} {af:<14} {al:<10} {same}")

positional_matches = sum(
    1 for c, a in zip(clean_rows, adv_rows)
    if c["filename"] == a["filename"] and c["label"] == a["label"]
)
print(f"\nPositional matches: {positional_matches}/{min(len(clean_rows), len(adv_rows))}")

# ---------------------------------------------------------------------------
# 3. Pixel-level check: find the clean image matching adv[0] by content
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("3. Pixel search: find adv[0] in clean archive by content")
print(f"{'='*60}")

adv0_fname = adv_rows[0]["filename"]
adv0_arr   = to_arr(adv_imgs[adv0_fname])
print(f"Adv[0] filename: {adv0_fname}, label: {adv_rows[0]['label']}")
print(f"Adv[0] pixel[0,0]: {adv0_arr[0,0,:]}, mean: {adv0_arr.mean():.2f}")

best_match_fname = None
best_match_diff  = float("inf")

# Search all clean images for the closest match to adv[0]
# (adv image has perturbation, so we allow up to ε=30 ~ 30 pixel units max diff)
for fname, raw in clean_imgs.items():
    c_arr = to_arr(raw)
    if c_arr.shape != adv0_arr.shape:
        continue
    diff = np.abs(c_arr.astype(float) - adv0_arr.astype(float)).max()
    if diff < best_match_diff:
        best_match_diff  = diff
        best_match_fname = fname
        if diff < 2:
            break  # exact match found

print(f"\nBest matching clean image: {best_match_fname}")
print(f"Max pixel diff to adv[0]: {best_match_diff:.1f}")

if best_match_diff < 15:
    print(f"✓ Found clean counterpart (diff={best_match_diff:.1f} ≤ 15 → likely same image + perturbation)")
    # Find its position in clean labels.csv
    for i, r in enumerate(clean_rows):
        if r["filename"] == best_match_fname:
            print(f"  Position in clean labels.csv: row {i}")
            print(f"  Clean label: {r['label']}, Adv label: {adv_rows[0]['label']}")
            break
else:
    print(f"✗ No close match found (best diff={best_match_diff:.1f})")
    print("  The clean archive may contain entirely different images than the adversarial archive.")

# ---------------------------------------------------------------------------
# 4. Check metadata.json if present
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("4. Check metadata.json in clean archive")
print(f"{'='*60}")
with open(clean_archive, "rb") as f:
    buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
buf.seek(0)
with tarfile.open(fileobj=buf, mode="r:") as tar:
    try:
        meta = tar.extractfile("metadata.json")
        import json
        d = json.load(meta)
        print(json.dumps(d, indent=2)[:500])
    except Exception as e:
        print(f"No metadata.json or error: {e}")