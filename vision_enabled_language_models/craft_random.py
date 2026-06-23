#!/usr/bin/env python3
"""
craft_random.py — Craft randomly-perturbed images (uniform Linf noise)
as a surrogate-free baseline for the RobustGenBench pipeline.

This mirrors craft_adversarial.py exactly:
  • reads clean archives from DATA_ROOT (downloaded from HF if missing)
  • writes perturbed PNGs to OUTPUT_ROOT/{run_name}/
  • packages to tar.zst under PACKAGED_ROOT/
  • uploads to HF_DATASET_REPO at:
        adversarial/random/linf_eps{N}/{dataset}_processed.tar.zst

The noise is sampled in uint8 space so that
    ||x_random - x_clean||_inf  <=  eps
holds BIT-EXACTLY on the stored bytes — no float/quantization drift.

    delta ~ U{-eps, ..., +eps}    (integer, per pixel, per channel)
    x_random = clip(x_clean + delta, 0, 255)            # uint8

Usage:
    # Linf 30/255, all datasets, craft + package + upload
    python craft_random.py --norm Linf --eps 30 --package --upload_hf

    # Single dataset, no upload (just inspect output)
    python craft_random.py --dataset caltech101 --norm Linf --eps 30

    # Different budget
    python craft_random.py --norm Linf --eps 8 --package --upload_hf

Seeding: each image gets a deterministic seed derived from
(global_seed, dataset, filename), so repeated runs produce identical
archives and all downstream models see the same noise.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths (mirrors craft_adversarial.py exactly)
# ---------------------------------------------------------------------------

TMP_ROOT        = Path("/tmp/robustgenbench")
DATA_ROOT       = Path(os.path.expanduser("~/links/scratch/robustgenbench/data_processed"))
HF_CACHE_DIR    = Path(os.path.expanduser("~/links/scratch/robustgenbench/hf_cache"))
OUTPUT_ROOT     = TMP_ROOT / "adversarial_examples"
PACKAGED_ROOT   = TMP_ROOT / "adversarial_packaged"
WORK_DIR        = Path(os.path.expanduser("~/links/scratch/robustgenbench/work"))

HF_DATASET_REPO = "MaxHeuillet/RobustGenBench"
CLASS_NAMES_DIR = DATA_ROOT / "class_names"

REAL_HF_HOME    = os.path.expanduser("~/.cache/huggingface")

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]


# ---------------------------------------------------------------------------
# Naming — parallels threat_model_slug / hf_archive_path from craft_adversarial.py
# ---------------------------------------------------------------------------

def eps_slug(eps: float) -> str:
    if float(eps) == int(float(eps)):
        return str(int(float(eps)))
    return str(float(eps)).replace(".", "_")


def threat_model_slug_random(norm: str, eps: float) -> str:
    """e.g. 'linf_eps30_random_uniform'."""
    return f"{norm.lower()}_eps{eps_slug(eps)}_random_uniform"


def run_dir_name(dataset: str, norm: str, eps: float) -> str:
    """e.g. 'caltech101__random__linf_eps30_random_uniform'."""
    return f"{dataset}__random__{threat_model_slug_random(norm, eps)}"


def hf_archive_path(norm: str, eps: float, archive_filename: str) -> str:
    """
    Path within the HF repo.
    Sits parallel to the surrogate folders:
        adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/*.tar.zst
        adversarial/random/linf_eps30_random_uniform/*.tar.zst
    """
    return f"adversarial/random/{threat_model_slug_random(norm, eps)}/{archive_filename}"


# ---------------------------------------------------------------------------
# Data download (copy of craft_adversarial.py)
# ---------------------------------------------------------------------------

def ensure_data_downloaded(force: bool = False):
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    sentinel = DATA_ROOT / ".download_complete"
    if sentinel.exists() and not force:
        print(f"Data already present at {DATA_ROOT}.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub"); sys.exit(1)

    print(f"\nDownloading {HF_DATASET_REPO!r} → {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
        cache_dir=str(HF_CACHE_DIR),
        ignore_patterns=["adversarial/*"],
    )
    sentinel.touch()
    print(f"\nDownload complete. Data stored at {DATA_ROOT}\n")


# ---------------------------------------------------------------------------
# Extract clean archive (same as craft_adversarial.py)
# ---------------------------------------------------------------------------

def extract_archive(dataset_name: str) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard"); sys.exit(1)

    archive_path = DATA_ROOT / f"{dataset_name}_processed.tar.zst"
    dest_dir     = WORK_DIR  / dataset_name

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if (dest_dir / "test" / "labels.csv").exists():
        return dest_dir

    print(f"Extracting {archive_path.name} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=dest_dir)
    return dest_dir


def load_local_dataset(dataset_dir: Path, split: str,
                        max_samples: Optional[int] = None):
    split_dir = dataset_dir / split
    csv_path  = split_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")
    items = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            items.append((split_dir / row["filename"], int(row["label"])))
    if max_samples is not None and max_samples < len(items):
        rng     = np.random.RandomState(42)
        indices = rng.choice(len(items), size=max_samples, replace=False)
        items   = [items[i] for i in sorted(indices)]
    return items


def load_class_names(dataset_name: str) -> dict:
    path = CLASS_NAMES_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Class names not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Per-image perturbation (uint8-exact)
# ---------------------------------------------------------------------------

def seed_for(dataset: str, filename: str, global_seed: int) -> int:
    h = hashlib.sha256(
        f"{global_seed}|{dataset}|{filename}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def perturb_image(img_path: Path, eps: int, seed: int) -> tuple[Image.Image, int]:
    """Uniform Linf noise in uint8 space. Returns (pil_image, actual_linf)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    rng   = np.random.default_rng(seed)
    noise = rng.integers(low=-eps, high=eps + 1, size=arr.shape, dtype=np.int16)

    perturbed = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    linf = int(np.max(np.abs(perturbed.astype(np.int16) - arr.astype(np.int16))))
    assert linf <= eps, f"Linf budget violated: {linf} > {eps}"

    return Image.fromarray(perturbed, mode="RGB"), linf


# ---------------------------------------------------------------------------
# Packaging (copy of craft_adversarial.py package_run)
# ---------------------------------------------------------------------------

def package_run(adv_dir: Path, output_dir: Path) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard"); sys.exit(1)

    meta_path = adv_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {adv_dir}")

    all_records = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]

    seen, records = set(), []
    for rec in all_records:
        fname = rec["image_path"]
        if fname not in seen:
            seen.add(fname)
            records.append(rec)

    if len(all_records) != len(records):
        print(f"  ⚠ Deduplicated metadata: {len(all_records)} → {len(records)} records")

    archive_name = f"{adv_dir.name}_processed.tar.zst"
    archive_path = output_dir / archive_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        print(f"  Archive already exists: {archive_path.name} — skipping packaging")
        return archive_path

    print(f"  Packaging {len(records)} images → {archive_path.name}")
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "label"])
    writer.writeheader()
    for rec in records:
        writer.writerow({"filename": rec["image_path"], "label": rec["label_idx"]})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    cctx = zstd.ZstdCompressor(level=3)
    with open(archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                csv_info      = tarfile.TarInfo(name="test/labels.csv")
                csv_info.size = len(csv_bytes)
                tar.addfile(csv_info, io.BytesIO(csv_bytes))
                for rec in tqdm(records, desc="  Compressing", leave=False):
                    img_path = adv_dir / rec["image_path"]
                    if img_path.exists():
                        tar.add(str(img_path), arcname=f"test/{rec['image_path']}")

    size_mb = archive_path.stat().st_size / 1e6
    print(f"  ✓ {archive_path.name} ({size_mb:.1f} MB)")
    return archive_path


# ---------------------------------------------------------------------------
# HF upload (copy of craft_adversarial.py, adapted path)
# ---------------------------------------------------------------------------

def upload_to_hf(archive_path: Path, norm: str, eps: float):
    """Upload to HF, explicitly passing the token to avoid HF_HOME confusion."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub"); sys.exit(1)

    # Read token directly from the real user cache — don't rely on env vars
    token_path = Path(REAL_HF_HOME) / "token"
    if not token_path.exists():
        print(f"  ✗ No token at {token_path} — run `huggingface-cli login`")
        sys.exit(1)
    token = token_path.read_text().strip()

    path_in_repo = hf_archive_path(norm, eps, archive_path.name)
    print(f"  Uploading → {HF_DATASET_REPO}/{path_in_repo}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  ✓ Upload complete: {archive_path.name}")


# ---------------------------------------------------------------------------
# Per-dataset runner (mirrors run_dataset / run_dataset_common)
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args):
    norm       = args.norm
    eps        = int(args.eps)                    # integer uint8 budget
    rname      = run_dir_name(dataset, norm, eps)
    output_dir = OUTPUT_ROOT / rname

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset}")
    print(f"  Norm    : {norm}  eps={eps}/255  (uint8-exact uniform noise)")
    print(f"  Seed    : {args.seed}")
    print(f"{'='*60}")

    # Skip if already completed
    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping")
        if args.package or args.upload_hf:
            archive_path = package_run(output_dir, PACKAGED_ROOT)
            if args.upload_hf:
                upload_to_hf(archive_path, norm, eps)
        return

    dataset_dir   = extract_archive(dataset)
    items         = load_local_dataset(dataset_dir, split="test",
                                        max_samples=args.max_samples)
    label_to_name = load_class_names(dataset)
    print(f"  {len(items)} samples | {len(label_to_name)} classes")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset": dataset,
        "norm":    norm,
        "eps":     eps,
        "eps_unit": "uint8  (i.e. eps/255 in [0,1] space)",
        "noise_distribution": "uniform integer in [-eps, +eps] per pixel per channel",
        "seed":    args.seed,
        "attack":  "random_uniform_linf_uint8",
    }, indent=2))

    n_total      = 0
    max_linf_obs = 0
    meta_file    = open(output_dir / "metadata.jsonl", "a")

    for img_path, label in tqdm(items, desc=dataset):
        seed = seed_for(dataset, img_path.name, args.seed)
        perturbed, linf = perturb_image(img_path, eps, seed)

        out = output_dir / Path(img_path).with_suffix(".png").name
        out.parent.mkdir(parents=True, exist_ok=True)
        perturbed.save(out, format="PNG")

        meta_file.write(json.dumps({
            "image_path": out.name,
            "label_idx":  int(label),
            "label_name": label_to_name.get(int(label), "unknown"),
            "linf_observed": linf,
        }) + "\n")
        meta_file.flush()
        n_total      += 1
        max_linf_obs  = max(max_linf_obs, linf)

    meta_file.close()
    (output_dir / "surrogate_summary.json").write_text(json.dumps({
        "n_total": n_total,
        "surrogate_clean_acc": None,
        "surrogate_adv_acc":   None,
        "attack_success_rate": None,
        "max_linf_observed":   max_linf_obs,
        "eps_budget":          eps,
        "note": "random uniform Linf noise — no surrogate used",
    }, indent=2))

    print(f"\n  {n_total} images saved | max Linf observed = {max_linf_obs}/{eps}")
    if args.package or args.upload_hf:
        archive_path = package_run(output_dir, PACKAGED_ROOT)
        if args.upload_hf:
            upload_to_hf(archive_path, norm, eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm",           default="Linf", choices=["Linf"],
                   help="Only Linf supported for random uniform noise.")
    p.add_argument("--eps",            type=int, default=30,
                   help="Linf budget in uint8 units (default 30 = 30/255)")
    p.add_argument("--dataset",        default=None, choices=ALL_DATASETS)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--seed",           type=int, default=0,
                   help="Global seed (combined with dataset + filename for per-image seeding)")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--package",        action="store_true")
    p.add_argument("--upload_hf",      action="store_true")
    args = p.parse_args()

    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, OUTPUT_ROOT, PACKAGED_ROOT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ensure_data_downloaded(force=args.force_download)

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for dataset in datasets:
        try:
            run_dataset(dataset, args)
        except Exception as e:
            print(f"\n  ERROR on {dataset}: {e}\n")
            import traceback; traceback.print_exc()

    print("\nAll done!")


if __name__ == "__main__":
    main()