#!/usr/bin/env python3
"""
package_adversarial.py — Repackage adversarial examples into the same format
as the original processed datasets, so llm_classify.py can use them directly.

Original format (what llm_classify.py expects):
  <dataset>_processed.tar.zst
    └── test/
        ├── labels.csv   (columns: filename, label)
        └── <image files>

Usage:
    # Single run
    python package_adversarial.py \
        --adv_dir /tmp/adversarial_examples/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard \
        --output_dir /tmp/adversarial_packaged

    # All runs in a folder
    python package_adversarial.py \
        --adv_root /tmp/adversarial_examples \
        --output_dir /tmp/adversarial_packaged
"""

import argparse
import csv
import io
import json
import os
import tarfile
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


def package_run(adv_dir: Path, output_dir: Path):
    """
    Convert one adversarial run directory into a _processed.tar.zst archive
    compatible with llm_classify.py.
    """
    meta_path = adv_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"  ⚠  No metadata.jsonl found in {adv_dir} — skipping")
        return

    records = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    if not records:
        print(f"  ⚠  Empty metadata.jsonl in {adv_dir} — skipping")
        return

    # Archive name = run_name_processed.tar.zst
    run_name     = adv_dir.name
    archive_name = f"{run_name}_processed.tar.zst"
    archive_path = output_dir / archive_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        print(f"  ✓ Already packaged — {archive_path}")
        return

    print(f"\n  Packaging {run_name}")
    print(f"  → {archive_path}")
    print(f"  Images: {len(records)}")

    # Build labels.csv in memory
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "label"])
    writer.writeheader()
    for rec in records:
        writer.writerow({"filename": rec["image_path"], "label": rec["label_idx"]})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # Write tar.zst
    cctx = zstd.ZstdCompressor(level=3)
    with open(archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:

                # Add labels.csv
                csv_info          = tarfile.TarInfo(name="test/labels.csv")
                csv_info.size     = len(csv_bytes)
                tar.addfile(csv_info, io.BytesIO(csv_bytes))

                # Add images
                for rec in tqdm(records, desc="  Compressing", leave=False):
                    img_path = adv_dir / rec["image_path"]
                    if not img_path.exists():
                        print(f"    ⚠  Missing image: {img_path}")
                        continue
                    tar.add(str(img_path), arcname=f"test/{rec['image_path']}")

    size_mb = archive_path.stat().st_size / 1e6
    print(f"  ✓ Done — {size_mb:.1f} MB")

    # Also write a companion class_names.json alongside the archive
    # so llm_classify.py --class_names_dir can find it
    cfg_path = adv_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f"  Dataset : {cfg.get('dataset', '?')}  eps={cfg.get('eps_pixel', '?')}/255")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--adv_dir",  type=str,
                   help="Single adversarial run directory to package")
    g.add_argument("--adv_root", type=str,
                   help="Root folder containing multiple run directories")
    p.add_argument("--output_dir", default="/tmp/adversarial_packaged",
                   help="Where to write the .tar.zst archives")
    args = p.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))

    if args.adv_dir:
        dirs = [Path(os.path.expanduser(args.adv_dir))]
    else:
        root = Path(os.path.expanduser(args.adv_root))
        dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and (d / "metadata.jsonl").exists()])
        print(f"Found {len(dirs)} run(s) to package")

    for adv_dir in dirs:
        try:
            package_run(adv_dir, output_dir)
        except Exception as e:
            print(f"  ERROR on {adv_dir.name}: {e}")
            continue

    print(f"\nAll archives saved to: {output_dir}")
    print("\nTo use with llm_classify.py:")
    print(f"  python llm_classify.py --batch --provider openai --model gpt-4o-mini \\")
    print(f"      --dataset <run_name> \\")
    print(f"      --data_root {output_dir} \\")
    print(f"      --class_names_dir ~/data_processed/class_names \\")
    print(f"      --run_name <run_name>__openai")


if __name__ == "__main__":
    main()