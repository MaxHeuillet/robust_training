#!/usr/bin/env python3
"""
verify_and_repack.py — Verify all ViT-H/14 Linf8 archives have exactly
1000 test images. Repack and re-upload any that are wrong.

Run on the cluster:
    python verify_and_repack.py
"""

import csv, io, json, os, sys, tarfile
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)

HF_REPO      = "MaxHeuillet/RobustGenBench"
SURROGATE    = "zeroshot_clip_vith14_laion2b"
TM_SLUG      = "linf_eps8_autoattack_standard"
ADV_ROOT     = Path(os.path.expanduser("~/data/adversarial"))
OUTPUT_ROOT  = Path("/tmp/robustgenbench/adversarial_examples")
PACKAGED_ROOT= Path("/tmp/robustgenbench/adversarial_packaged")
REAL_HF_HOME = os.path.expanduser("~/.cache/huggingface")

DATASET_SIZES = {
    "caltech101":                 1000,
    "fgvc-aircraft-2013b":        1000,
    "flowers-102":                1000,
    "oxford-iiit-pet":            1000,
    "stanford_cars":              1000,
    "uc-merced-land-use-dataset": 420,
}

DATASETS = list(DATASET_SIZES.keys())

def read_archive_size(archive_path: Path) -> tuple[int, list[str]]:
    """Returns (n_images, filenames_from_labels_csv)."""
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        members = tar.getmembers()
        n_imgs  = sum(1 for m in members if m.name.endswith(".png"))
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f    = tar.extractfile(tar.getmember(cand))
                rows = list(csv.DictReader(io.TextIOWrapper(f)))
                return n_imgs, [r["filename"] for r in rows]
            except KeyError:
                continue
    return n_imgs, []


CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")

def get_canonical_test_filenames(dataset: str) -> list[tuple[str, str]]:
    """
    Load the canonical (filename, label) pairs from the CLEAN archive's
    test/labels.csv. These are the ground-truth 1000 test images.
    Falls back to first `expected` rows of the adversarial archive if
    clean archive is not available.
    """
    clean_archive = CLEAN_ROOT / f"{dataset}_processed.tar.zst"
    if not clean_archive.exists():
        print(f"  ⚠ Clean archive not found at {clean_archive} — will use adv labels.csv order")
        return []

    with open(clean_archive, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f    = tar.extractfile(tar.getmember(cand))
                rows = list(csv.DictReader(io.TextIOWrapper(f)))
                result = [(r["filename"], r["label"]) for r in rows]
                print(f"  Clean archive labels.csv: {len(result)} entries")
                return result
            except KeyError:
                continue
    return []


def repack_from_archive(dataset: str, expected: int) -> Path | None:
    """
    Repack directly from the existing (wrong) archive, keeping only the
    images whose filenames match the canonical test set from the clean archive.
    This guarantees we keep the right images, not arbitrary first-N rows.
    """
    archive_name = f"{dataset}__{SURROGATE}__{TM_SLUG}_processed.tar.zst"
    src_path     = ADV_ROOT / SURROGATE / TM_SLUG / archive_name

    if not src_path.exists():
        print(f"  ⚠ Source archive not found: {src_path}")
        return None

    # Get canonical test filenames from clean archive
    canonical = get_canonical_test_filenames(dataset)
    if not canonical:
        print(f"  ⚠ Could not load canonical test filenames — aborting repack")
        return None

    canonical_fnames = {fname for fname, _ in canonical}
    print(f"  Canonical test set: {len(canonical_fnames)} unique filenames")

    # Load source archive
    print(f"  Reading source archive ({src_path.stat().st_size//1_000_000} MB)...")
    with open(src_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)

    with tarfile.open(fileobj=buf, mode="r:") as tar:
        # Read adv labels.csv to get label mapping
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f    = tar.extractfile(tar.getmember(cand))
                adv_rows = list(csv.DictReader(io.TextIOWrapper(f)))
                break
            except KeyError:
                continue

        adv_label_map = {r["filename"]: r["label"] for r in adv_rows}

        # Load only images that belong to the canonical test set
        imgs = {}
        extra = []
        for member in tar.getmembers():
            if not (member.name.endswith(".png") or member.name.endswith(".jpg")):
                continue
            fname = Path(member.name).name
            if fname in canonical_fnames:
                imgs[fname] = tar.extractfile(member).read()
            else:
                extra.append(fname)

    print(f"  Found {len(imgs)}/{len(canonical_fnames)} canonical images in archive")
    if extra:
        print(f"  Excluded {len(extra)} non-test images: {extra[:5]}{'...' if len(extra)>5 else ''}")

    missing = canonical_fnames - set(imgs.keys())
    if missing:
        print(f"  ⚠ Missing {len(missing)} canonical images: {sorted(missing)[:5]}")

    # Build output rows in canonical order, with labels from adv archive
    rows = []
    for fname, clean_label in canonical:
        if fname in imgs:
            # Use adv label (should match clean label — sanity check)
            adv_label = adv_label_map.get(fname, clean_label)
            if adv_label != clean_label:
                print(f"  ⚠ Label mismatch for {fname}: clean={clean_label} adv={adv_label}")
            rows.append({"filename": fname, "label": adv_label})

    print(f"  Final archive will contain {len(rows)} images")

    # Write corrected labels.csv
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "label"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # Write corrected archive
    PACKAGED_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PACKAGED_ROOT / archive_name
    if out_path.exists():
        out_path.unlink()

    print(f"  Writing corrected archive → {archive_name}")
    cctx = zstd.ZstdCompressor(level=3)
    with open(out_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar_out:
                csv_info      = tarfile.TarInfo(name="test/labels.csv")
                csv_info.size = len(csv_bytes)
                tar_out.addfile(csv_info, io.BytesIO(csv_bytes))
                for r in rows:
                    fname = r["filename"]
                    raw   = imgs.get(fname)
                    if raw:
                        info      = tarfile.TarInfo(name=f"test/{fname}")
                        info.size = len(raw)
                        tar_out.addfile(info, io.BytesIO(raw))
                    else:
                        print(f"  ⚠ Missing image for {fname}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✓ {archive_name} ({size_mb:.1f} MB, {len(rows)} images)")
    return out_path
    """
    Rebuild a clean archive from metadata.jsonl, keeping only the first
    `expected` unique filenames (by filename, deduped, sorted by index).
    """
    adv_dir_name = f"{dataset}__{SURROGATE}__{TM_SLUG}"
    adv_dir      = OUTPUT_ROOT / adv_dir_name

    if not adv_dir.exists():
        print(f"  ⚠ Output dir not found: {adv_dir}")
        return None

    meta_path = adv_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"  ⚠ metadata.jsonl not found in {adv_dir}")
        return None

    all_records = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]

    # Deduplicate by filename, keep first occurrence
    seen, deduped = set(), []
    for rec in all_records:
        fname = rec["image_path"]
        if fname not in seen:
            seen.add(fname)
            deduped.append(rec)

    if len(deduped) > expected:
        print(f"  Trimming {len(deduped)} → {expected} records")
        deduped = deduped[:expected]
    elif len(deduped) < expected:
        print(f"  ⚠ Only {len(deduped)} unique records, expected {expected}")

    # Write clean labels.csv
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "label"])
    writer.writeheader()
    for rec in deduped:
        writer.writerow({"filename": rec["image_path"], "label": rec["label_idx"]})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # Package
    archive_name = f"{adv_dir_name}_processed.tar.zst"
    PACKAGED_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = PACKAGED_ROOT / archive_name

    # Force repack
    if archive_path.exists():
        archive_path.unlink()
        print(f"  Removed stale archive")

    print(f"  Repacking {len(deduped)} images → {archive_name}")
    cctx = zstd.ZstdCompressor(level=3)
    with open(archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                csv_info      = tarfile.TarInfo(name="test/labels.csv")
                csv_info.size = len(csv_bytes)
                tar.addfile(csv_info, io.BytesIO(csv_bytes))
                for rec in deduped:
                    img_path = adv_dir / rec["image_path"]
                    if img_path.exists():
                        tar.add(str(img_path), arcname=f"test/{rec['image_path']}")
                    else:
                        print(f"  ⚠ Missing image: {img_path}")

    size_mb = archive_path.stat().st_size / 1e6
    print(f"  ✓ {archive_name} ({size_mb:.1f} MB)")
    return archive_path


def upload(archive_path: Path):
    from huggingface_hub import HfApi
    os.environ["HF_HOME"] = REAL_HF_HOME
    path_in_repo = f"adversarial/{SURROGATE}/{TM_SLUG}/{archive_path.name}"
    print(f"  Uploading → {HF_REPO}/{path_in_repo}")
    HfApi().upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"  ✓ Upload complete")


# ---------------------------------------------------------------------------
# Main — verify all datasets
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  Verifying ViT-H/14 Linf8 archives")
print(f"{'='*60}\n")

to_repack = []

for dataset in DATASETS:
    expected = DATASET_SIZES[dataset]
    archive_path = ADV_ROOT / SURROGATE / TM_SLUG / \
        f"{dataset}__{SURROGATE}__{TM_SLUG}_processed.tar.zst"

    if not archive_path.exists():
        print(f"  ✗ {dataset:<35} NOT FOUND locally")
        continue

    n_imgs, csv_rows = read_archive_size(archive_path)
    n_csv = len(csv_rows)
    ok    = n_csv == expected and n_imgs == expected

    status = "✓ OK" if ok else f"✗ WRONG ({n_imgs} imgs, {n_csv} csv rows, expected {expected})"
    print(f"  {status:<12} {dataset}")

    if not ok:
        to_repack.append(dataset)

if not to_repack:
    print("\n  All archives correct — nothing to do.")
    sys.exit(0)

print(f"\n  Need to repack: {to_repack}")
print()

for dataset in to_repack:
    expected = DATASET_SIZES[dataset]
    print(f"\n  Repacking {dataset}...")
    archive_path = repack_from_archive(dataset, expected)
    if archive_path:
        upload(archive_path)
        local_path = ADV_ROOT / SURROGATE / TM_SLUG / archive_path.name
        import shutil
        shutil.copy2(archive_path, local_path)
        print(f"  ✓ Local copy updated: {local_path}")

print("\n  Done.")