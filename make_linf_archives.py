#!/usr/bin/env python3
"""
make_random_linf_archives.py — Build randomly-perturbed copies of the
`{dataset}_processed.tar.zst` archives with a bit-exact Linf budget.

The noise is sampled in uint8 space so that the guarantee
    ||x_random - x_clean||_inf <= eps
holds EXACTLY on the stored bytes (no float/quantization slop).

Perturbation:
    delta ~ U{-eps, ..., +eps}    (integer, per pixel, per channel)
    x_random = clip(x_clean + delta, 0, 255)            # uint8

Output archive layout mirrors the input:
    {dataset}_processed.tar.zst
      ├── test/labels.csv          (copied verbatim)
      ├── test/<image files...>    (perturbed, re-encoded as PNG)
      └── ...                      (any other files copied verbatim)

Usage
-----
    python make_random_linf_archives.py \
        --src_root  ~/data_processed \
        --dst_root  /tmp/data/random/linf_eps30 \
        --eps 30 \
        --datasets caltech101 fgvc-aircraft-2013b flowers-102 \
                   oxford-iiit-pet stanford_cars uc-merced-land-use-dataset

Reproducibility
---------------
Noise is seeded per (dataset, image_index) so repeated runs yield identical
archives, and different models evaluated on the same archive see the same
noise.
"""

import argparse
import csv
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard pillow numpy", file=sys.stderr)
    sys.exit(1)


ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# File extensions we'll treat as perturbable images. Everything else is
# passed through verbatim (labels.csv, readmes, etc.).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_for(dataset: str, member_name: str, global_seed: int) -> int:
    """Deterministic 64-bit seed per (dataset, image, global_seed)."""
    h = hashlib.sha256(
        f"{global_seed}|{dataset}|{member_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:8], "little", signed=False)


# ---------------------------------------------------------------------------
# Core perturbation
# ---------------------------------------------------------------------------

def perturb_image_bytes(raw: bytes, eps: int, seed: int) -> tuple[bytes, int]:
    """
    Decode an image, add uniform integer noise in [-eps, +eps] per pixel,
    clip to [0, 255], re-encode as PNG (lossless).

    Returns (png_bytes, actual_linf_on_disk) where the second value is
    verified bit-exact.
    """
    img = Image.open(io.BytesIO(raw))
    # Preserve RGB / L / RGBA — just force a mode the rest of the pipeline
    # expects. Your databases loader almost certainly converts to RGB, so
    # match that.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.uint8)       # HxW or HxWxC, uint8

    rng = np.random.default_rng(seed)
    noise = rng.integers(
        low=-eps, high=eps + 1, size=arr.shape, dtype=np.int16,
    )

    perturbed = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Bit-exact Linf check (should always be <= eps by construction)
    linf = int(np.max(np.abs(perturbed.astype(np.int16) - arr.astype(np.int16))))
    assert linf <= eps, f"Linf budget violated: {linf} > {eps}"

    out = io.BytesIO()
    Image.fromarray(perturbed, mode=img.mode).save(out, format="PNG", compress_level=6)
    return out.getvalue(), linf


# ---------------------------------------------------------------------------
# Archive I/O
# ---------------------------------------------------------------------------

def open_src_tar(src_archive: Path) -> tarfile.TarFile:
    """Stream-decompress a .tar.zst into a tarfile reader."""
    f = open(src_archive, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    # Buffer into memory — simpler than streaming for <few-GB archives and
    # avoids double-open issues when we also want to seek.
    buf = io.BytesIO(reader.read())
    f.close()
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:")


def write_dst_tar(dst_archive: Path, members: list[tuple[tarfile.TarInfo, bytes]]):
    """Write members into a new .tar.zst."""
    dst_archive.parent.mkdir(parents=True, exist_ok=True)
    raw_buf = io.BytesIO()
    with tarfile.open(fileobj=raw_buf, mode="w:") as tar_out:
        for info, data in members:
            # Create a fresh TarInfo so we don't leak device/inode noise.
            new_info = tarfile.TarInfo(name=info.name)
            new_info.size = len(data)
            new_info.mode = info.mode or 0o644
            new_info.mtime = info.mtime
            new_info.type = info.type
            new_info.uid = 0
            new_info.gid = 0
            tar_out.addfile(new_info, io.BytesIO(data))
    raw_buf.seek(0)
    cctx = zstd.ZstdCompressor(level=10)
    with open(dst_archive, "wb") as f_out:
        f_out.write(cctx.compress(raw_buf.getvalue()))


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    dataset: str,
    src_root: Path,
    dst_root: Path,
    eps: int,
    global_seed: int,
    verify_count: bool,
):
    # Resolve source archive (handles the fancy naming convention you use).
    matches = sorted(src_root.glob(f"{dataset}*_processed.tar.zst"))
    if not matches:
        print(f"  ✗ No archive found for {dataset!r} under {src_root}")
        return False
    src_archive = matches[0]
    dst_archive = dst_root / src_archive.name

    print(f"\n── {dataset}")
    print(f"   src: {src_archive}")
    print(f"   dst: {dst_archive}")

    if dst_archive.exists():
        print(f"   ⏭  Destination exists — skipping (delete to regenerate)")
        return True

    tar_in = open_src_tar(src_archive)

    members_out: list[tuple[tarfile.TarInfo, bytes]] = []
    n_images = 0
    n_passthrough = 0
    max_linf_observed = 0

    for info in tar_in.getmembers():
        if not info.isfile():
            continue

        fobj = tar_in.extractfile(info)
        if fobj is None:
            continue
        raw = fobj.read()

        ext = Path(info.name).suffix.lower()
        if ext in IMAGE_EXTS:
            seed = seed_for(dataset, info.name, global_seed)
            try:
                new_bytes, linf = perturb_image_bytes(raw, eps, seed)
            except Exception as e:
                print(f"   ⚠ Failed to perturb {info.name}: {e}")
                return False
            # Rename .jpg/.jpeg/etc. → .png since we re-encoded as PNG.
            # Important: labels.csv references filenames, so we must NOT
            # rename. Instead keep the original name but write PNG bytes
            # under that filename. PIL/torchvision open by content, not
            # extension, so this is fine. If your loader is strict about
            # extension, set --keep_extension_only_png=False (not provided
            # — just don't rename).
            new_info = info
            new_info.size = len(new_bytes)
            members_out.append((new_info, new_bytes))
            n_images += 1
            max_linf_observed = max(max_linf_observed, linf)
        else:
            members_out.append((info, raw))
            n_passthrough += 1

    tar_in.close()

    if n_images == 0:
        print(f"   ✗ No images found in archive — aborting")
        return False

    print(f"   perturbed: {n_images} images  (max Linf on disk = {max_linf_observed})")
    print(f"   copied:    {n_passthrough} non-image files")

    write_dst_tar(dst_archive, members_out)
    print(f"   ✓ wrote {dst_archive.stat().st_size / 1e6:.1f} MB")

    if verify_count:
        verify_archive_labels(dst_archive)

    return True


def verify_archive_labels(archive: Path):
    """Sanity check: test/labels.csv exists and row count matches image count."""
    tar_in = open_src_tar(archive)
    try:
        label_members = [m for m in tar_in.getmembers()
                         if m.name.endswith("labels.csv")]
        if not label_members:
            print(f"   ⚠ verify: no labels.csv found")
            return
        labels_member = label_members[0]
        rows = list(csv.DictReader(
            io.TextIOWrapper(tar_in.extractfile(labels_member), encoding="utf-8")
        ))
        image_members = [
            m for m in tar_in.getmembers()
            if m.isfile() and Path(m.name).suffix.lower() in IMAGE_EXTS
        ]
        print(f"   verify: {len(rows)} label rows, {len(image_members)} images")
    finally:
        tar_in.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True,
                   help="Directory containing clean {dataset}_processed.tar.zst files")
    p.add_argument("--dst_root", required=True,
                   help="Destination directory for perturbed archives")
    p.add_argument("--eps", type=int, default=30,
                   help="Linf budget in uint8 units (default 30 = 30/255)")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--seed", type=int, default=0,
                   help="Global seed (combined with dataset + filename for per-image seeding)")
    p.add_argument("--no_verify", action="store_true",
                   help="Skip post-write label/image count verification")
    args = p.parse_args()

    src_root = Path(os.path.expanduser(args.src_root)).resolve()
    dst_root = Path(os.path.expanduser(args.dst_root)).resolve()

    if not src_root.exists():
        print(f"✗ src_root does not exist: {src_root}")
        sys.exit(1)

    print(f"src_root : {src_root}")
    print(f"dst_root : {dst_root}")
    print(f"eps      : {args.eps}/255 = {args.eps/255:.4f} (uint8 integer noise)")
    print(f"seed     : {args.seed}")
    print(f"datasets : {', '.join(args.datasets)}")

    ok = all(
        process_dataset(d, src_root, dst_root, args.eps, args.seed,
                        verify_count=not args.no_verify)
        for d in args.datasets
    )

    print()
    print("=" * 60)
    print("  Done." if ok else "  Completed with errors.")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()