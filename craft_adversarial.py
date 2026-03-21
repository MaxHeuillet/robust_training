#!/usr/bin/env python3
"""
craft_adversarial.py — Craft adversarial perturbations using AutoAttack
against either:
  • zero-shot CLIP ViT-B/16   (open_clip, LAION-2B weights)
  • zero-shot SigLIP2-base-patch16-224  (HuggingFace transformers)

Supports Linf, L1, L2 norms.

Usage:
    # CLIP, Linf, all datasets
    python craft_adversarial.py --surrogate clip --norm Linf --eps 30 --batch_size 8

    # CLIP, L2, all datasets
    python craft_adversarial.py --surrogate clip --norm L2 --eps 1 --batch_size 8

    # SigLIP2, L1, single dataset
    python craft_adversarial.py --surrogate siglip2 --norm L1 --eps 12 --dataset flowers-102 --batch_size 8

    # Upload to HuggingFace after crafting
    python craft_adversarial.py --surrogate siglip2 --norm Linf --eps 30 --upload_hf

    # Force re-download
    python craft_adversarial.py --surrogate siglip2 --force_download
"""

import argparse
import csv
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from autoattack import AutoAttack
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cluster-friendly /tmp paths
# ---------------------------------------------------------------------------

TMP_ROOT       = Path("/tmp/robustgenbench")
DATA_ROOT      = TMP_ROOT / "data_processed"
HF_CACHE_DIR   = TMP_ROOT / "hf_cache"
OUTPUT_ROOT    = TMP_ROOT / "adversarial_examples"
WORK_DIR       = TMP_ROOT / "work"

HF_DATASET_REPO = "MaxHeuillet/RobustGenBench"
CLASS_NAMES_DIR = DATA_ROOT / "class_names"


# ---------------------------------------------------------------------------
# Surrogate identifiers
# ---------------------------------------------------------------------------

SURROGATE_CLIP   = "clip"
SURROGATE_SIGLIP = "siglip2"
ALL_SURROGATES   = [SURROGATE_CLIP, SURROGATE_SIGLIP]

CLIP_MODEL    = "ViT-B-16"
CLIP_PRETRAIN = "laion2b_s34b_b88k"
CLIP_MEAN     = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_MEAN     = (0.5, 0.5, 0.5)
SIGLIP_STD      = (0.5, 0.5, 0.5)

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# Supported norms and their AutoAttack string
ALL_NORMS = ["Linf", "L2", "L1"]

# Default epsilon values per norm (in pixel units 0-255)
DEFAULT_EPS = {
    "Linf": 30,
    "L2":   1000,   # ~3.9 in [0,1] for 224x224 images
    "L1":   2000,
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Warning: no GPU found, using CPU — this will be slow")
    return device


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def surrogate_slug(surrogate: str) -> str:
    if surrogate == SURROGATE_CLIP:
        return "zeroshot_clip_vitb16_laion2b"
    elif surrogate == SURROGATE_SIGLIP:
        return "zeroshot_siglip2_base_patch16_224"
    raise ValueError(surrogate)


def threat_model_slug(norm: str, eps: int) -> str:
    """e.g. 'linf_eps30_autoattack_standard'"""
    return f"{norm.lower()}_eps{eps}_autoattack_standard"


def run_name(dataset: str, surrogate: str, norm: str, eps: int) -> str:
    return f"{dataset}__{surrogate_slug(surrogate)}__{threat_model_slug(norm, eps)}"


def hf_path(surrogate: str, norm: str, eps: int, filename: str) -> str:
    """
    Returns the path_in_repo for HuggingFace upload.
    Structure:
      adversarial/<surrogate_slug>/<threat_model_slug>/<filename>
    """
    return f"adversarial/{surrogate_slug(surrogate)}/{threat_model_slug(norm, eps)}/{filename}"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def ensure_data_downloaded(force: bool = False):
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    sentinel = DATA_ROOT / ".download_complete"
    if sentinel.exists() and not force:
        print(f"Data already present at {DATA_ROOT} (use --force_download to re-fetch).")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    print(f"\nDownloading {HF_DATASET_REPO!r} → {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
        cache_dir=str(HF_CACHE_DIR),
        # Only download clean data, not adversarial archives
        ignore_patterns=["adversarial/*"],
    )
    sentinel.touch()
    print(f"\nDownload complete. Data stored at {DATA_ROOT}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_archive(dataset_name: str) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard")
        sys.exit(1)

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


def load_local_dataset(dataset_dir: Path, split: str, max_samples: Optional[int] = None):
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
# Dataset wrapper
# ---------------------------------------------------------------------------

class AdversarialDataset(Dataset):
    def __init__(self, items: list, transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label, img_path.name


# ---------------------------------------------------------------------------
# Surrogate models
# ---------------------------------------------------------------------------

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model, text_features: torch.Tensor, device, temperature: float = 100.0):
        super().__init__()
        self.clip_model  = clip_model
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(CLIP_STD,  device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        image_features = self.clip_model.encode_image(x)
        image_features = F.normalize(image_features, dim=-1)
        return self.temperature * (image_features @ self.text_features.T)


class ZeroShotSigLIP2(nn.Module):
    def __init__(self, encode_fn, text_features: torch.Tensor, device, temperature: float = 100.0):
        super().__init__()
        self._encode_fn  = encode_fn
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(SIGLIP_STD,  device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        image_features = self._encode_fn(x)
        image_features = F.normalize(image_features, dim=-1)
        return self.temperature * (image_features @ self.text_features.T)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DATASET_PROMPT_TEMPLATES = {
    "caltech101":                 lambda name: f"a photo of a {name}",
    "fgvc-aircraft-2013b":        lambda name: f"a photo of a {name}, a type of aircraft",
    "flowers-102":                lambda name: f"a photo of a {name}, a type of flower",
    "oxford-iiit-pet":            lambda name: f"a photo of a {name}, a type of pet",
    "stanford_cars":              lambda name: f"a photo of a {name}",
    "uc-merced-land-use-dataset": lambda name: f"a satellite photo of a {name}",
}
DEFAULT_PROMPT_TEMPLATE = lambda name: f"a photo of a {name}"


def build_prompts(dataset: str, class_names: list) -> list:
    template = DATASET_PROMPT_TEMPLATES.get(dataset, DEFAULT_PROMPT_TEMPLATE)
    return [template(name) for name in class_names]


def load_clip_surrogate(label_to_name: dict, device: torch.device, dataset: str = "") -> ZeroShotCLIP:
    import open_clip
    print(f"\nLoading CLIP surrogate: {CLIP_MODEL} / {CLIP_PRETRAIN}")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, cache_dir=str(HF_CACHE_DIR),
    )
    clip_model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer(CLIP_MODEL)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        tokens        = tokenizer(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)
    model = ZeroShotCLIP(clip_model, text_features, device)
    model.eval().to(device)
    return model


def load_siglip2_surrogate(label_to_name: dict, device: torch.device, dataset: str = "") -> ZeroShotSigLIP2:
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel
    print(f"\nLoading SigLIP2 surrogate: {SIGLIP_MODEL_ID}")
    vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR))
    vision_model.eval().to(device)
    text_model = SiglipTextModel.from_pretrained(SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR))
    text_model.eval().to(device)
    tokenizer   = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR), use_fast=False)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        max_len = text_model.config.max_position_embeddings
        inputs  = tokenizer(prompts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
        text_outputs  = text_model(**inputs)
        text_features = F.normalize(text_outputs.pooler_output, dim=-1)
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return vision_model(pixel_values=x).pooler_output

    model = ZeroShotSigLIP2(encode_fn, text_features, device)
    model.eval().to(device)
    return model


def load_surrogate(surrogate: str, label_to_name: dict, device: torch.device, dataset: str = "") -> nn.Module:
    if surrogate == SURROGATE_CLIP:
        return load_clip_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_SIGLIP:
        return load_siglip2_surrogate(label_to_name, device, dataset=dataset)
    raise ValueError(f"Unknown surrogate: {surrogate!r}")


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Save adversarial images
# ---------------------------------------------------------------------------

def save_batch(x_adv, labels, filenames, label_to_name, output_dir: Path) -> list[dict]:
    to_pil  = T.ToPILImage()
    records = []
    for img_t, label, fname in zip(x_adv, labels, filenames):
        out = output_dir / Path(fname).with_suffix(".png").name
        out.parent.mkdir(parents=True, exist_ok=True)
        to_pil(img_t.clamp(0, 1).cpu()).save(out, format="PNG")
        records.append({
            "image_path": out.name,
            "label_idx":  int(label),
            "label_name": label_to_name.get(int(label), "unknown"),
        })
    return records


# ---------------------------------------------------------------------------
# Package into tar.zst (same format as clean datasets)
# ---------------------------------------------------------------------------

def package_run(adv_dir: Path, output_dir: Path) -> Path:
    """Convert adversarial PNG folder → _processed.tar.zst archive."""
    import io
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard")
        sys.exit(1)

    meta_path = adv_dir / "metadata.jsonl"
    records   = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]

    archive_name = f"{adv_dir.name}_processed.tar.zst"
    archive_path = output_dir / archive_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        print(f"  Archive already exists: {archive_path}")
        return archive_path

    print(f"  Packaging {len(records)} images → {archive_path.name}")

    # Build labels.csv in memory
    import csv as csv_mod
    csv_buf = io.StringIO()
    writer  = csv_mod.DictWriter(csv_buf, fieldnames=["filename", "label"])
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
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hf(archive_path: Path, surrogate: str, norm: str, eps: int):
    from huggingface_hub import HfApi
    api = HfApi()
    path_in_repo = hf_path(surrogate, norm, eps, archive_path.name)
    print(f"  Uploading → {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  ✓ Uploaded {archive_path.name}")


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args, device: torch.device):
    norm      = args.norm
    eps       = args.eps
    eps_float = eps / 255.0 if norm == "Linf" else eps / 255.0

    # For L1/L2 the epsilon is typically specified differently.
    # We keep eps in pixel units for the slug but convert to float for AutoAttack.
    # User should pass sensible values per norm (see DEFAULT_EPS).
    rname      = run_name(dataset, args.surrogate, norm, eps)
    output_dir = OUTPUT_ROOT / rname

    print(f"\n{'='*60}")
    print(f"  Dataset   : {dataset}")
    print(f"  Surrogate : {args.surrogate}")
    print(f"  Norm      : {norm}  eps={eps}/255={eps_float:.5f}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*60}")

    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping\n")
        return

    dataset_dir   = extract_archive(dataset)
    items         = load_local_dataset(dataset_dir, split="test", max_samples=args.max_samples)
    label_to_name = load_class_names(dataset)
    print(f"Loaded {len(items)} samples | {len(label_to_name)} classes")

    model = load_surrogate(args.surrogate, label_to_name, device, dataset=dataset)

    transform = build_transform()
    ds        = AdversarialDataset(items, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset":        dataset,
        "surrogate":      args.surrogate,
        "surrogate_slug": surrogate_slug(args.surrogate),
        "norm":           norm,
        "eps_pixel":      eps,
        "eps_float":      eps_float,
        "attack":         "autoattack_standard",
    }, indent=2))

    adversary = AutoAttack(
        model, norm=norm, eps=eps_float,
        version="standard",
        device=device, verbose=True,
    )

    n_correct_clean = n_correct_adv = n_total = 0
    meta_file = open(output_dir / "metadata.jsonl", "a")
    print(f"\nCrafting adversarial examples → {output_dir}\n")

    for x, labels, filenames in tqdm(loader, desc=dataset):
        x        = x.to(device)
        labels_t = labels.to(device) if isinstance(labels, torch.Tensor) \
                   else torch.tensor(labels, dtype=torch.long).to(device)

        with torch.no_grad():
            n_correct_clean += (model(x).argmax(1) == labels_t).sum().item()

        x_adv = adversary.run_standard_evaluation(x, labels_t, bs=x.size(0))

        with torch.no_grad():
            n_correct_adv += (model(x_adv).argmax(1) == labels_t).sum().item()

        n_total += x.size(0)

        records = save_batch(x_adv, labels_t.cpu().tolist(), filenames, label_to_name, output_dir)
        for rec in records:
            meta_file.write(json.dumps(rec) + "\n")
        meta_file.flush()

    meta_file.close()

    clean_acc = n_correct_clean / n_total
    adv_acc   = n_correct_adv   / n_total
    (output_dir / "surrogate_summary.json").write_text(json.dumps({
        "n_total":             n_total,
        "surrogate_clean_acc": round(clean_acc, 4),
        "surrogate_adv_acc":   round(adv_acc,   4),
        "attack_success_rate": round(1 - adv_acc, 4),
    }, indent=2))

    print(f"\n  Surrogate clean acc  : {clean_acc:.4f}")
    print(f"  Surrogate adv acc    : {adv_acc:.4f}")
    print(f"  Attack success rate  : {1 - adv_acc:.4f}")

    # Package and optionally upload
    if args.package or args.upload_hf:
        packaged_dir = TMP_ROOT / "adversarial_packaged"
        archive_path = package_run(output_dir, packaged_dir)
        if args.upload_hf:
            upload_to_hf(archive_path, args.surrogate, norm, eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate",      default=SURROGATE_CLIP, choices=ALL_SURROGATES)
    p.add_argument("--norm",           default="Linf", choices=ALL_NORMS,
                   help="Threat model norm: Linf, L2, L1")
    p.add_argument("--eps",            type=int, default=None,
                   help="Epsilon in pixel units [0-255]. Defaults: Linf=30, L2=1000, L1=2000")
    p.add_argument("--dataset",        default=None)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--package",        action="store_true",
                   help="Package adversarial PNGs into tar.zst after crafting")
    p.add_argument("--upload_hf",      action="store_true",
                   help="Package and upload to HuggingFace after crafting (implies --package)")
    args = p.parse_args()

    # Apply default epsilon if not specified
    if args.eps is None:
        args.eps = DEFAULT_EPS[args.norm]
        print(f"Using default eps={args.eps} for norm={args.norm}")

    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, OUTPUT_ROOT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ensure_data_downloaded(force=args.force_download)

    device   = get_device()
    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    print(f"\nSurrogate : {args.surrogate}")
    print(f"Norm      : {args.norm}  eps={args.eps}")
    print(f"Datasets  : {', '.join(datasets)}")
    print(f"Output    : {OUTPUT_ROOT}")

    for dataset in datasets:
        try:
            run_dataset(dataset, args, device)
        except Exception as e:
            print(f"\n  ERROR on {dataset}: {e} — skipping\n")
            continue

    print("\nAll done!")


if __name__ == "__main__":
    main()