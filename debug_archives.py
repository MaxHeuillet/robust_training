#!/usr/bin/env python3
"""
Quick diagnostic: print the first 5 filenames from labels.csv
in both the clean and adversarial archive for caltech101,
and show image shapes + first pixel values.
"""
import csv, io, os, sys, tarfile
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)

CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")
ADV_ROOT   = Path(os.path.expanduser("~/data/adversarial"))

def peek_archive(archive_path: Path, label: str):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  {archive_path}")
    print(f"{'─'*60}")
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        members = tar.getmembers()
        print(f"  Total members: {len(members)}")
        print(f"  First 8 members: {[m.name for m in members[:8]]}")

        # Read labels.csv
        try:
            csv_f = tar.extractfile(tar.getmember("test/labels.csv"))
            rows  = list(csv.DictReader(io.TextIOWrapper(csv_f)))
            print(f"  labels.csv rows: {len(rows)}")
            print(f"  First 3 rows: {rows[:3]}")
        except Exception as e:
            print(f"  labels.csv error: {e}")
            rows = []

        # Load first image
        if rows:
            fname = rows[0]["filename"]
            try:
                raw = tar.extractfile(f"test/{fname}").read()
                img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                print(f"  First image shape : {img.shape}")
                print(f"  First image dtype : {img.dtype}")
                print(f"  First pixel RGB   : {img[0,0,:]}")
                print(f"  Mean pixel value  : {img.mean():.2f}")
            except Exception as e:
                print(f"  Image load error: {e}")

# Clean
clean_archive = CLEAN_ROOT / "caltech101_processed.tar.zst"
peek_archive(clean_archive, "CLEAN — caltech101")

# Adversarial Linf eps8
adv_folder = ADV_ROOT / "zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"
adv_archives = sorted(adv_folder.glob("caltech101*_processed.tar.zst"))
if adv_archives:
    peek_archive(adv_archives[0], "ADV — caltech101 Linf eps8")
else:
    print("\nNo adversarial archive found for caltech101 Linf eps8")