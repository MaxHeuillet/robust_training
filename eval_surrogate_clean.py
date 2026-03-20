#!/usr/bin/env python3
"""
eval_surrogate_clean.py — Measure zero-shot clean accuracy for all 5 surrogate
models across all 6 benchmark datasets.

Surrogates:
  1. CLIP ViT-B/16      (open_clip, LAION-2B)
  2. SigLIP2 base       (google/siglip2-base-patch16-224)
  3. CLIP ViT-H/14      (open_clip, OpenAI)
  4. SigLIP2 SO400M     (google/siglip2-so400m-patch14-384, bicubic upsample to 384)
  5. DINOv2 ViT-L/14    (facebook/dinov2-large, cosine sim over text features via OpenCLIP text encoder)

All paths are /tmp-based for cluster compatibility.
Data is downloaded automatically from MaxHeuillet/RobustGenBench if not present.

Usage:
    python eval_surrogate_clean.py
    python eval_surrogate_clean.py --surrogate clip_vitb16 --dataset caltech101
    python eval_surrogate_clean.py --max_samples 200
    python eval_surrogate_clean.py --output_json results/surrogate_clean.json
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# /tmp paths
# ---------------------------------------------------------------------------

TMP_ROOT        = Path("/tmp/robustgenbench")
DATA_ROOT       = TMP_ROOT / "data_processed"
HF_CACHE_DIR    = TMP_ROOT / "hf_cache"
WORK_DIR        = TMP_ROOT / "work"
CLASS_NAMES_DIR = DATA_ROOT / "class_names"
HF_DATASET_REPO = "MaxHeuillet/RobustGenBench"


# ---------------------------------------------------------------------------
# Surrogate registry
# ---------------------------------------------------------------------------

ALL_SURROGATES = [
    "clip_vitb16",
    "siglip2_base",
    "clip_vith14",
    "siglip2_so400m",
    "dinov2_vitl14",
]

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# Dataset-specific prompt templates
PROMPT_TEMPLATES = {
    "caltech101":                 lambda n: f"a photo of a {n}",
    "fgvc-aircraft-2013b":        lambda n: f"a photo of a {n}, a type of aircraft",
    "flowers-102":                lambda n: f"a photo of a {n}, a type of flower",
    "oxford-iiit-pet":            lambda n: f"a photo of a {n}, a type of pet",
    "stanford_cars":              lambda n: f"a photo of a {n}",
    "uc-merced-land-use-dataset": lambda n: f"a satellite photo of {n}",
}
DEFAULT_TEMPLATE = lambda n: f"a photo of a {n}"

def build_prompts(dataset: str, class_names: list) -> list:
    tmpl = PROMPT_TEMPLATES.get(dataset, DEFAULT_TEMPLATE)
    return [tmpl(name) for name in class_names]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Warning: no GPU found, using CPU")
    return device


# ---------------------------------------------------------------------------
# Data download
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
    print(f"Downloading {HF_DATASET_REPO} → {DATA_ROOT} (ignoring adversarial folder)...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset",
        local_dir=str(DATA_ROOT), cache_dir=str(HF_CACHE_DIR),
        ignore_patterns="adversarial/*",
    )
    sentinel.touch()
    print("Download complete.")


def extract_archive(dataset_name: str) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard"); sys.exit(1)
    archive_path = DATA_ROOT / f"{dataset_name}_processed.tar.zst"
    dest_dir     = WORK_DIR / dataset_name
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if (dest_dir / "test" / "labels.csv").exists():
        return dest_dir
    print(f"  Extracting {archive_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=dest_dir)
    return dest_dir


def load_local_dataset(dataset_dir: Path, split: str, max_samples: Optional[int] = None):
    split_dir = dataset_dir / split
    csv_path  = split_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append((split_dir / row["filename"], int(row["label"])))
    if max_samples and max_samples < len(items):
        rng     = np.random.RandomState(42)
        indices = rng.choice(len(items), size=max_samples, replace=False)
        items   = [items[i] for i in sorted(indices)]
    return items


def load_class_names(dataset_name: str) -> dict:
    path = CLASS_NAMES_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Class names not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    def __init__(self, items: list, transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Surrogate model wrappers
# All models expect raw [0,1] input and output cosine-similarity logits.
# ---------------------------------------------------------------------------

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model, text_features, device, mean, std, temperature=100.0):
        super().__init__()
        self.clip_model  = clip_model
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std,  device=device).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = F.normalize(self.clip_model.encode_image(x), dim=-1)
        return self.temperature * (feats @ self.text_features.T)


class ZeroShotSigLIP2(nn.Module):
    """
    Wraps encode_fn as a plain Python callable (not nn.Module child) to avoid
    AutoAttack's model serialiser chocking on HuggingFace PreTrainedModel configs.
    For SO400M the encode_fn upsamples to 384×384 internally before encoding.
    """
    def __init__(self, encode_fn, text_features, device, temperature=100.0):
        super().__init__()
        self._encode_fn  = encode_fn
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1))

    def forward(self, x):
        x    = (x - self.mean) / self.std
        feats = F.normalize(self._encode_fn(x), dim=-1)
        return self.temperature * (feats @ self.text_features.T)


class ZeroShotDINOv2(nn.Module):
    """
    DINOv2 has no text encoder. We use OpenCLIP's ViT-L/14 text encoder
    (OpenAI weights) to produce text embeddings, and DINOv2 ViT-L/14 for image
    embeddings. Both produce 1024-dim features — compatible for cosine similarity.
    """
    def __init__(self, dino_model, text_features, device, temperature=100.0):
        super().__init__()
        self.dino_model  = dino_model
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        # DINOv2 uses ImageNet normalisation
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1))

    def forward(self, x):
        x    = (x - self.mean) / self.std
        feats = self.dino_model(x)           # (B, 1024)
        feats = F.normalize(feats, dim=-1)
        return self.temperature * (feats @ self.text_features.T)


# ---------------------------------------------------------------------------
# Surrogate loaders
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def load_clip_vitb16(label_to_name, dataset, device):
    import open_clip
    print("  Loading CLIP ViT-B/16 (LAION-2B)...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", cache_dir=str(HF_CACHE_DIR))
    model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer("ViT-B-16")
    class_names = [label_to_name[i] for i in sorted(label_to_name)]
    prompts     = build_prompts(dataset, class_names)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_f = F.normalize(model.encode_text(tokens), dim=-1)
    return ZeroShotCLIP(model, text_f, device, CLIP_MEAN, CLIP_STD)


def load_clip_vith14(label_to_name, dataset, device):
    import open_clip
    print("  Loading CLIP ViT-H/14 (OpenAI)...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=str(HF_CACHE_DIR))
    model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer("ViT-H-14")
    class_names = [label_to_name[i] for i in sorted(label_to_name)]
    prompts     = build_prompts(dataset, class_names)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_f = F.normalize(model.encode_text(tokens), dim=-1)
    return ZeroShotCLIP(model, text_f, device, CLIP_MEAN, CLIP_STD)


def _load_siglip2(model_id, label_to_name, dataset, device, upsample_to=None):
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel
    print(f"  Loading {model_id}...")
    vision_model = SiglipVisionModel.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    vision_model.eval().to(device)
    text_model   = SiglipTextModel.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    text_model.eval().to(device)
    tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR), use_fast=False)
    class_names  = [label_to_name[i] for i in sorted(label_to_name)]
    prompts      = build_prompts(dataset, class_names)
    max_len      = text_model.config.max_position_embeddings
    with torch.no_grad():
        inputs = tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        text_f = F.normalize(text_model(**inputs).pooler_output, dim=-1)
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if upsample_to:
        # Perturbations crafted at 224×224, surrogate forward pass at 384×384
        def encode_fn(x):
            x_up = F.interpolate(x, size=(upsample_to, upsample_to),
                                 mode="bicubic", align_corners=False)
            return vision_model(pixel_values=x_up).pooler_output
    else:
        def encode_fn(x):
            return vision_model(pixel_values=x).pooler_output

    return ZeroShotSigLIP2(encode_fn, text_f, device)


def load_siglip2_base(label_to_name, dataset, device):
    return _load_siglip2(
        "google/siglip2-base-patch16-224", label_to_name, dataset, device)


def load_siglip2_so400m(label_to_name, dataset, device):
    return _load_siglip2(
        "google/siglip2-so400m-patch14-384", label_to_name, dataset, device,
        upsample_to=384)


def load_dinov2_vitl14(label_to_name, dataset, device):
    import open_clip
    print("  Loading DINOv2 ViT-L/14 + CLIP ViT-L/14 text encoder...")

    # Image encoder: DINOv2 ViT-L/14 (outputs 1024-dim CLS token)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                           source="github")
    dino.eval().to(device)

    # Text encoder: OpenCLIP ViT-L/14 (also 768-dim) — use projection to 1024
    # Actually ViT-L/14 in open_clip outputs 768-dim text features.
    # DINOv2 ViT-L/14 outputs 1024-dim. We need matching dims.
    # Solution: use ViT-H/14 text encoder which outputs 1024-dim to match DINOv2.
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=str(HF_CACHE_DIR))
    clip_model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer("ViT-H-14")
    class_names = [label_to_name[i] for i in sorted(label_to_name)]
    prompts     = build_prompts(dataset, class_names)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_f = F.normalize(clip_model.encode_text(tokens), dim=-1)

    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ZeroShotDINOv2(dino, text_f, device)


SURROGATE_LOADERS = {
    "clip_vitb16":    load_clip_vitb16,
    "siglip2_base":   load_siglip2_base,
    "clip_vith14":    load_clip_vith14,
    "siglip2_so400m": load_siglip2_so400m,
    "dinov2_vitl14":  load_dinov2_vitl14,
}

SURROGATE_LABELS = {
    "clip_vitb16":    "CLIP ViT-B/16 (LAION-2B)",
    "siglip2_base":   "SigLIP2 base-patch16-224",
    "clip_vith14":    "CLIP ViT-H/14 (OpenAI)",
    "siglip2_so400m": "SigLIP2 SO400M-patch14-384",
    "dinov2_vitl14":  "DINOv2 ViT-L/14",
}


# ---------------------------------------------------------------------------
# Shared image transform — 224×224 for all surrogates.
# SO400M upsamples internally in its encode_fn.
# ---------------------------------------------------------------------------

def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels in loader:
            x      = x.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels, dtype=torch.long).to(device)
            preds  = model(x).argmax(1)
            correct += (preds == labels).sum().item()
            total   += x.size(0)
    return correct, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate",      default=None, choices=ALL_SURROGATES,
                   help="Single surrogate to evaluate (default: all)")
    p.add_argument("--dataset",        default=None, choices=ALL_DATASETS,
                   help="Single dataset (default: all)")
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--max_samples",    type=int, default=None,
                   help="Cap per dataset (default: full test set)")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--output_json",    default=None,
                   help="Path to save results as JSON")
    args = p.parse_args()

    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ensure_data_downloaded(force=args.force_download)

    device     = get_device()
    surrogates = [args.surrogate] if args.surrogate else ALL_SURROGATES
    datasets   = [args.dataset]   if args.dataset   else ALL_DATASETS
    transform  = build_transform()
    results    = {}

    col_w = max(len(SURROGATE_LABELS[s]) for s in surrogates) + 2
    ds_w  = max(len(d) for d in datasets) + 2

    print(f"\n{'Surrogate':<{col_w}}  {'Dataset':<{ds_w}}  {'Accuracy':>10}  {'Correct':>12}")
    print("─" * (col_w + ds_w + 30))

    for surrogate in surrogates:
        results[surrogate] = {}
        loader_fn = SURROGATE_LOADERS[surrogate]

        for dataset in datasets:
            try:
                label_to_name = load_class_names(dataset)
                model         = loader_fn(label_to_name, dataset, device)
                model.eval().to(device)

                dataset_dir = extract_archive(dataset)
                items       = load_local_dataset(dataset_dir, "test", args.max_samples)
                ds          = EvalDataset(items, transform)
                loader      = DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

                correct, total = evaluate(model, loader, device)
                acc = correct / total

                results[surrogate][dataset] = {
                    "accuracy": round(acc, 4),
                    "correct":  correct,
                    "total":    total,
                }

                label = SURROGATE_LABELS[surrogate]
                print(f"{label:<{col_w}}  {dataset:<{ds_w}}  {acc:>10.4f}  {correct:>6}/{total:<6}")

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR [{surrogate}][{dataset}]: {e}")
                results.setdefault(surrogate, {})[dataset] = {"error": str(e)}

    # Summary
    print("\n" + "═" * (col_w + ds_w + 30))
    print(f"{'Surrogate':<{col_w}}  {'Avg accuracy':>12}")
    print("─" * (col_w + 16))
    for surrogate in surrogates:
        accs = [v["accuracy"] for v in results[surrogate].values() if "accuracy" in v]
        if accs:
            print(f"{SURROGATE_LABELS[surrogate]:<{col_w}}  {sum(accs)/len(accs):>12.4f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()