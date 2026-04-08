#!/usr/bin/env python3
"""
craft_adversarial.py — Craft adversarial / corrupted images using AutoAttack
or common corruptions (ImageNet-C style), against either:
  • zero-shot CLIP ViT-B/16   (open_clip, LAION-2B weights)
  • zero-shot CLIP ViT-H/14   (open_clip, LAION-2B weights)
  • zero-shot MetaCLIP ViT-H/14 (open_clip, fullcc2.5b weights)  ← NEW
  • zero-shot SigLIP2-base-patch16-224  (HuggingFace transformers)

Supports Linf, L1, L2 norms (gradient-based) and common corruptions.

Epsilon conventions (matching AutoAttack's [0,1] input space):
  Linf : specified in pixel units /255   e.g. --eps 30  → 30/255
  L2   : specified directly              e.g. --eps 2.0 → 2.0
  L1   : specified directly              e.g. --eps 75  → 75.0
  common: no epsilon — severity controlled by --severity (1-5, default 3)

Usage:
    python craft_adversarial.py --surrogate metaclip_h14 --norm Linf --eps 8  --upload_hf
    python craft_adversarial.py --surrogate metaclip_h14 --norm Linf --eps 30 --upload_hf
    python craft_adversarial.py --surrogate metaclip_h14 --norm L2   --eps 2  --upload_hf
    python craft_adversarial.py --surrogate metaclip_h14 --norm L2   --eps 8  --upload_hf
    python craft_adversarial.py --surrogate metaclip_h14 --norm L1   --eps 75 --upload_hf
    python craft_adversarial.py --surrogate metaclip_h14 --norm L1   --eps 300 --upload_hf
"""

import argparse
import csv
import io
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
# /tmp paths
# ---------------------------------------------------------------------------

TMP_ROOT        = Path("/tmp/robustgenbench")
DATA_ROOT       = Path(os.path.expanduser("~/links/scratch/robustgenbench/data_processed"))
HF_CACHE_DIR    = Path(os.path.expanduser("~/links/scratch/robustgenbench/hf_cache"))
OUTPUT_ROOT     = TMP_ROOT / "adversarial_examples"
PACKAGED_ROOT   = TMP_ROOT / "adversarial_packaged"
WORK_DIR        = Path(os.path.expanduser("~/links/scratch/robustgenbench/work"))

HF_DATASET_REPO = "MaxHeuillet/RobustGenBench"
CLASS_NAMES_DIR = DATA_ROOT / "class_names"

# Real HF home for token lookup (not /tmp)
REAL_HF_HOME    = os.path.expanduser("~/.cache/huggingface")


# ---------------------------------------------------------------------------
# Surrogate identifiers
# ---------------------------------------------------------------------------

SURROGATE_CLIP       = "clip"
SURROGATE_SIGLIP     = "siglip2"
SURROGATE_CLIP_H     = "clip_vith14"
SURROGATE_METACLIP_H = "metaclip_h14"
SURROGATE_SIGLIP_SO400M    = "siglip2_so400m"
SURROGATE_SIGLIP_SO400M_384 = "siglip2_so400m_384"

ALL_SURROGATES = [SURROGATE_CLIP, SURROGATE_SIGLIP, SURROGATE_CLIP_H,
                  SURROGATE_METACLIP_H, SURROGATE_SIGLIP_SO400M,
                  SURROGATE_SIGLIP_SO400M_384]


# CLIP ViT-B/16
CLIP_MODEL      = "ViT-B-16"
CLIP_PRETRAIN   = "laion2b_s34b_b88k"
CLIP_MEAN       = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD        = (0.26862954, 0.26130258, 0.27577711)

# CLIP ViT-H/14
CLIP_H_MODEL    = "ViT-H-14"
CLIP_H_PRETRAIN = "laion2b_s32b_b79k"
# ViT-H/14 uses same normalization as ViT-B/16 (OpenCLIP default)

# MetaCLIP ViT-H/14 fullcc2.5b
METACLIP_H_MODEL    = "ViT-H-14-quickgelu"
METACLIP_H_PRETRAIN = "metaclip_fullcc"
# MetaCLIP uses the same OpenAI CLIP normalization constants

# SigLIP2
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_MEAN     = (0.5, 0.5, 0.5)
SIGLIP_STD      = (0.5, 0.5, 0.5)

SIGLIP_SO400M_MODEL_ID     = "google/siglip2-so400m-patch16-naflex"
SIGLIP_SO400M_SIZE = 224  # input size; patchify upsamples to correct patch grid internally
SIGLIP_SO400M_MEAN         = (0.5, 0.5, 0.5)
SIGLIP_SO400M_STD          = (0.5, 0.5, 0.5)

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

ALL_NORMS = ["Linf", "L2", "L1", "common"]

DEFAULT_SEVERITY = 3

CORRUPTION_TYPES = [
    "shot_noise", "impulse_noise", "defocus_blur", "motion_blur",
    "zoom_blur", "snow", "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]


# ---------------------------------------------------------------------------
# Epsilon handling
# ---------------------------------------------------------------------------

def eps_to_float(norm: str, eps: float) -> float:
    if norm == "Linf":
        return eps / 255.0
    return float(eps)


def eps_slug(eps: float) -> str:
    if float(eps) == int(float(eps)):
        return str(int(float(eps)))
    return str(float(eps)).replace(".", "_")


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def surrogate_slug(surrogate: str) -> str:
    if surrogate == SURROGATE_CLIP:
        return "zeroshot_clip_vitb16_laion2b"
    elif surrogate == SURROGATE_SIGLIP:
        return "zeroshot_siglip2_base_patch16_224"
    elif surrogate == SURROGATE_CLIP_H:
        return "zeroshot_clip_vith14_laion2b"
    elif surrogate == SURROGATE_METACLIP_H:
        return "zeroshot_metaclip_vith14_fullcc2_5b"
    elif surrogate == SURROGATE_SIGLIP_SO400M:
        return "zeroshot_siglip2_so400m_patch16_naflex"
    elif surrogate == SURROGATE_SIGLIP_SO400M_384:
        return "zeroshot_siglip2_so400m_patch14_384"
    raise ValueError(surrogate)


def threat_model_slug(norm: str, eps: float = 0, severity: int = DEFAULT_SEVERITY) -> str:
    if norm == "common":
        return f"common_severity{severity}"
    return f"{norm.lower()}_eps{eps_slug(eps)}_autoattack_standard"


def run_dir_name(dataset: str, surrogate: str, norm: str, eps: float = 0,
                 severity: int = DEFAULT_SEVERITY) -> str:
    if norm == "common":
        return f"{dataset}__common_severity{severity}"
    return f"{dataset}__{surrogate_slug(surrogate)}__{threat_model_slug(norm, eps)}"


def hf_archive_path(surrogate: str, norm: str, eps: float,
                    archive_filename: str, severity: int = DEFAULT_SEVERITY) -> str:
    if norm == "common":
        return f"adversarial/common/{threat_model_slug(norm, severity=severity)}/{archive_filename}"
    return (
        f"adversarial/{surrogate_slug(surrogate)}"
        f"/{threat_model_slug(norm, eps)}"
        f"/{archive_filename}"
    )


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
# Data loading
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
# Surrogate model wrappers
# ---------------------------------------------------------------------------

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model, text_features: torch.Tensor, device,
                 mean=CLIP_MEAN, std=CLIP_STD, temperature: float = 100.0):
        super().__init__()
        self.clip_model  = clip_model
        self.temperature = temperature
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        image_features = F.normalize(self.clip_model.encode_image(x), dim=-1)
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
        image_features = F.normalize(self._encode_fn(x), dim=-1)
        return self.temperature * (image_features @ self.text_features.T)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DATASET_PROMPT_TEMPLATES = {
    "caltech101":                 lambda n: f"a photo of a {n}",
    "fgvc-aircraft-2013b":        lambda n: f"a photo of a {n}, a type of aircraft",
    "flowers-102":                lambda n: f"a photo of a {n}, a type of flower",
    "oxford-iiit-pet":            lambda n: f"a photo of a {n}, a type of pet",
    "stanford_cars":              lambda n: f"a photo of a {n}",
    "uc-merced-land-use-dataset": lambda n: f"a satellite photo of a {n}",
}
DEFAULT_PROMPT_TEMPLATE = lambda n: f"a photo of a {n}"

def build_prompts(dataset: str, class_names: list) -> list:
    tmpl = DATASET_PROMPT_TEMPLATES.get(dataset, DEFAULT_PROMPT_TEMPLATE)
    return [tmpl(name) for name in class_names]


# ---------------------------------------------------------------------------
# Surrogate loaders
# ---------------------------------------------------------------------------

def load_clip_surrogate(label_to_name: dict, device: torch.device,
                        dataset: str = "") -> ZeroShotCLIP:
    import open_clip
    print(f"\nLoading CLIP ViT-B/16 surrogate: {CLIP_MODEL} / {CLIP_PRETRAIN}")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, cache_dir=str(HF_CACHE_DIR))
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


def load_clip_h_surrogate(label_to_name: dict, device: torch.device,
                           dataset: str = "") -> ZeroShotCLIP:
    import open_clip
    print(f"\nLoading CLIP ViT-H/14 surrogate: {CLIP_H_MODEL} / {CLIP_H_PRETRAIN}")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_H_MODEL, pretrained=CLIP_H_PRETRAIN, cache_dir=str(HF_CACHE_DIR))
    clip_model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer(CLIP_H_MODEL)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        tokens        = tokenizer(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)
    # ViT-H/14 uses the same CLIP normalization constants
    model = ZeroShotCLIP(clip_model, text_features, device,
                         mean=CLIP_MEAN, std=CLIP_STD)
    model.eval().to(device)
    return model


def load_metaclip_h_surrogate(label_to_name: dict, device: torch.device,
                               dataset: str = "") -> ZeroShotCLIP:
    import open_clip
    print(f"\nLoading MetaCLIP ViT-H/14 surrogate: {METACLIP_H_MODEL} / {METACLIP_H_PRETRAIN}")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        METACLIP_H_MODEL, pretrained=METACLIP_H_PRETRAIN, cache_dir=str(HF_CACHE_DIR))
    clip_model.eval().to(device)
    tokenizer   = open_clip.get_tokenizer(METACLIP_H_MODEL)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        tokens        = tokenizer(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)
    # MetaCLIP uses the same CLIP normalization constants (OpenAI defaults)
    model = ZeroShotCLIP(clip_model, text_features, device,
                         mean=CLIP_MEAN, std=CLIP_STD)
    model.eval().to(device)
    return model


def load_siglip2_surrogate(label_to_name: dict, device: torch.device,
                            dataset: str = "") -> ZeroShotSigLIP2:
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel
    print(f"\nLoading SigLIP2 surrogate: {SIGLIP_MODEL_ID}")
    vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR))
    vision_model.eval().to(device)
    text_model   = SiglipTextModel.from_pretrained(SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR))
    text_model.eval().to(device)
    tokenizer    = AutoTokenizer.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR), use_fast=False)
    class_names  = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts      = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        max_len = text_model.config.max_position_embeddings
        inputs  = tokenizer(prompts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
        text_features = F.normalize(text_model(**inputs).pooler_output, dim=-1)
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return vision_model(pixel_values=x).pooler_output

    model = ZeroShotSigLIP2(encode_fn, text_features, device)
    model.eval().to(device)
    return model

def load_siglip2_so400m_surrogate(label_to_name: dict, device: torch.device,
                                   dataset: str = "") -> nn.Module:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model_id = "google/siglip2-so400m-patch16-naflex"
    print(f"\nLoading SigLIP2-SO400M-NaFlex surrogate: {model_id}")

    hf_model  = AutoModel.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    hf_model.eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))

    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")

    max_len = hf_model.config.text_config.max_position_embeddings
    with torch.no_grad():
        text_inputs  = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt"
        ).to(device)
        text_outputs = hf_model.text_model(**text_inputs)
        text_f       = text_outputs.pooler_output
        if hasattr(hf_model, 'text_projection') and hf_model.text_projection is not None:
            text_f = hf_model.text_projection(text_f)
        text_features = F.normalize(text_f, dim=-1)

    from PIL import Image
    import numpy as np
    dummy    = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    proc_out = processor(images=dummy, return_tensors="pt", padding="max_length")
    fixed_meta = {k: v.to(device) for k, v in proc_out.items()
                  if k != "pixel_values"}
    print(f"  Fixed meta keys: {list(fixed_meta.keys())}")
    _, N, patch_dim = proc_out["pixel_values"].shape
    print(f"  NaFlex: N={N} patches, patch_dim={patch_dim}")
    H = W = int(N ** 0.5) * 16

    # Replicate exactly what the processor does to pixel_values:
    # resize to 256×256, normalize with mean=0.5 std=0.5, then patchify.
    # This is differentiable so AutoAttack gradients flow through correctly.
    H = W = int(N ** 0.5) * 16  # 256 for this model

    class NaFlexSurrogate(nn.Module):
        def __init__(self, model, text_feats, meta, n, pdim, h, w, dev,
                     temperature=100.0):
            super().__init__()
            self._model      = model
            self.temperature = temperature
            self.n           = n
            self.pdim        = pdim
            self.h           = h
            self.w           = w
            self.register_buffer("text_features", text_feats)
            for k, v in meta.items():
                self.register_buffer(f"_meta_{k}", v)
            self._meta_keys = list(meta.keys())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            # Step 1: resize to 256×256
            x = F.interpolate(x, size=(self.h, self.w),
                            mode="bilinear", align_corners=False)  # bilinear, not bicubic
            # Step 2: normalize
            x = (x - 0.5) / 0.5
            # Step 3: patchify into (B, N, patch_dim) — H,W,C ordering
            x = x.unfold(2, 16, 16).unfold(3, 16, 16)
            x = x.permute(0, 2, 3, 4, 5, 1).contiguous()  # (B, nh, nw, ph, pw, C)
            x = x.reshape(B, self.n, self.pdim)
            # Step 4: expand fixed metadata to batch size and rename key
            meta = {}
            for k in self._meta_keys:
                v = getattr(self, f"_meta_{k}").expand(
                    B, *getattr(self, f"_meta_{k}").shape[1:])
                out_key = "attention_mask" if k == "pixel_attention_mask" else k
                meta[out_key] = v
            # Step 5: forward through vision model
            vision_out = self._model.vision_model(pixel_values=x, **meta)
            img_f = vision_out.pooler_output
            if hasattr(self._model, 'visual_projection') and \
            self._model.visual_projection is not None:
                img_f = self._model.visual_projection(img_f)
            img_f = F.normalize(img_f, dim=-1)
            return self.temperature * (img_f @ self.text_features.T)

    return NaFlexSurrogate(
        hf_model, text_features, fixed_meta, N, patch_dim, H, W, device
    ).to(device)

def load_siglip2_so400m_384_surrogate(label_to_name: dict, device: torch.device,
                                       dataset: str = "") -> ZeroShotSigLIP2:
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel
    model_id = "google/siglip2-so400m-patch14-384"
    print(f"\nLoading SigLIP2-SO400M-384 surrogate: {model_id}")
    vision_model = SiglipVisionModel.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    vision_model.eval().to(device)
    text_model   = SiglipTextModel.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR))
    text_model.eval().to(device)
    tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=str(HF_CACHE_DIR), use_fast=False)
    class_names  = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts      = build_prompts(dataset, class_names)
    print(f"  Encoding {len(prompts)} class prompts...")
    max_len = text_model.config.max_position_embeddings
    with torch.no_grad():
        text_inputs   = tokenizer(prompts, padding="max_length", truncation=True,
                                  max_length=max_len, return_tensors="pt").to(device)
        text_features = F.normalize(text_model(**text_inputs).pooler_output, dim=-1)
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=(384, 384), mode="bicubic", align_corners=False)
        return vision_model(pixel_values=x_up).pooler_output
    wrapper = ZeroShotSigLIP2(encode_fn, text_features, device)
    wrapper.eval().to(device)
    return wrapper

def load_surrogate(surrogate: str, label_to_name: dict,
                   device: torch.device, dataset: str = "") -> nn.Module:
    if surrogate == SURROGATE_CLIP:
        return load_clip_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_CLIP_H:
        return load_clip_h_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_METACLIP_H:
        return load_metaclip_h_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_SIGLIP:
        return load_siglip2_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_SIGLIP_SO400M:
        return load_siglip2_so400m_surrogate(label_to_name, device, dataset=dataset)
    elif surrogate == SURROGATE_SIGLIP_SO400M_384:
        return load_siglip2_so400m_384_surrogate(label_to_name, device, dataset=dataset)
    raise ValueError(f"Unknown surrogate: {surrogate!r}")


def build_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

def surrogate_img_size(surrogate: str) -> int:
    return 224  # all surrogates use 224×224 input; patchify handles NaFlex internally


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
# Package adversarial folder → tar.zst
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

    # Deduplicate by filename — metadata.jsonl can accumulate duplicates
    # across resumed runs. Keep first occurrence (earliest = correct).
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
# HuggingFace upload — fixed token lookup
# ---------------------------------------------------------------------------

def upload_to_hf(archive_path: Path, surrogate: str, norm: str,
                 eps: float = 0, severity: int = DEFAULT_SEVERITY):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub"); sys.exit(1)

    # Always use the real HF home for token lookup.
    # The script overrides HF_HOME to /tmp for model caching, but the login
    # token lives in ~/.cache/huggingface — restore it before uploading.
    os.environ["HF_HOME"] = REAL_HF_HOME

    path_in_repo = hf_archive_path(surrogate, norm, eps, archive_path.name, severity)
    print(f"  Uploading → {HF_DATASET_REPO}/{path_in_repo}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  ✓ Upload complete: {archive_path.name}")

    # Restore /tmp cache for any subsequent model downloads
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)


# ---------------------------------------------------------------------------
# Common corruptions runner
# ---------------------------------------------------------------------------

def apply_corruption(img_pil: Image.Image, corruption_name: str, severity: int) -> Image.Image:
    from corruptions.common import corrupt
    img_np = np.array(img_pil.convert("RGB"))
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    corrupted = corrupt(img_np, corruption_name=corruption_name, severity=severity)
    return Image.fromarray(np.uint8(corrupted))


def run_dataset_common(dataset: str, args):
    severity   = args.severity
    rname      = f"{dataset}__common_severity{severity}"
    output_dir = OUTPUT_ROOT / rname

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset}  |  common severity={severity}")
    print(f"{'='*60}")

    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping")
        if args.package or args.upload_hf:
            archive_path = package_run(output_dir, PACKAGED_ROOT)
            if args.upload_hf:
                upload_to_hf(archive_path, args.surrogate, "common", severity=severity)
        return

    dataset_dir   = extract_archive(dataset)
    items         = load_local_dataset(dataset_dir, split="test", max_samples=args.max_samples)
    label_to_name = load_class_names(dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset": dataset, "norm": "common", "severity": severity,
        "corruptions": CORRUPTION_TYPES,
    }, indent=2))

    n_total   = 0
    meta_file = open(output_dir / "metadata.jsonl", "a")

    for idx, (img_path, label) in enumerate(tqdm(items, desc=dataset)):
        corruption_name = CORRUPTION_TYPES[idx % len(CORRUPTION_TYPES)]
        img_pil         = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
        corrupted_pil   = apply_corruption(img_pil, corruption_name, severity)
        out = output_dir / Path(img_path).with_suffix(".png").name
        out.parent.mkdir(parents=True, exist_ok=True)
        corrupted_pil.save(out, format="PNG")
        meta_file.write(json.dumps({
            "image_path": out.name, "label_idx": int(label),
            "label_name": label_to_name.get(int(label), "unknown"),
            "corruption_type": corruption_name, "severity": severity,
        }) + "\n")
        meta_file.flush()
        n_total += 1

    meta_file.close()
    (output_dir / "surrogate_summary.json").write_text(json.dumps({
        "n_total": n_total, "surrogate_clean_acc": None,
        "surrogate_adv_acc": None, "attack_success_rate": None,
        "note": "common corruptions — no surrogate used",
    }, indent=2))

    print(f"  {n_total} images saved → {output_dir}")
    if args.package or args.upload_hf:
        archive_path = package_run(output_dir, PACKAGED_ROOT)
        if args.upload_hf:
            upload_to_hf(archive_path, args.surrogate, "common", severity=severity)


# ---------------------------------------------------------------------------
# Per-dataset runner (gradient-based attacks)
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args, device: torch.device):
    norm      = args.norm
    eps       = float(args.eps)
    eps_float = eps_to_float(norm, eps)
    rname      = run_dir_name(dataset, args.surrogate, norm, eps)
    output_dir = OUTPUT_ROOT / rname

    print(f"\n{'='*60}")
    print(f"  Dataset   : {dataset}")
    print(f"  Surrogate : {args.surrogate}  ({surrogate_slug(args.surrogate)})")
    print(f"  Norm      : {norm}  eps={eps}  (AA: {eps_float:.5f})")
    print(f"{'='*60}")

    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping")
        if args.package or args.upload_hf:
            archive_path = package_run(output_dir, PACKAGED_ROOT)
            if args.upload_hf:
                upload_to_hf(archive_path, args.surrogate, norm, eps)
        return

    dataset_dir   = extract_archive(dataset)
    items         = load_local_dataset(dataset_dir, split="test", max_samples=args.max_samples)
    label_to_name = load_class_names(dataset)
    print(f"  {len(items)} samples | {len(label_to_name)} classes")

    model = load_surrogate(args.surrogate, label_to_name, device, dataset=dataset)

    img_size  = surrogate_img_size(args.surrogate)
    transform = build_transform(size=img_size)

    ds        = AdversarialDataset(items, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset": dataset, "surrogate": args.surrogate,
        "surrogate_slug": surrogate_slug(args.surrogate),
        "norm": norm, "eps_user": eps, "eps_autoattack": eps_float,
        "attack": "autoattack_standard",
    }, indent=2))

    adversary = AutoAttack(
        model, norm=norm, eps=eps_float,
        version="standard", device=device, verbose=True,
    )

    n_correct_clean = n_correct_adv = n_total = 0
    meta_file = open(output_dir / "metadata.jsonl", "a")

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
        "n_total": n_total,
        "surrogate_clean_acc": round(clean_acc, 4),
        "surrogate_adv_acc":   round(adv_acc,   4),
        "attack_success_rate": round(1 - adv_acc, 4),
    }, indent=2))

    print(f"\n  {n_total} images | clean={clean_acc:.4f} | adv={adv_acc:.4f} | "
          f"success={1-adv_acc:.4f}")

    if args.package or args.upload_hf:
        archive_path = package_run(output_dir, PACKAGED_ROOT)
        if args.upload_hf:
            upload_to_hf(archive_path, args.surrogate, norm, eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate",      default=SURROGATE_CLIP, choices=ALL_SURROGATES)
    p.add_argument("--norm",           default="Linf", choices=ALL_NORMS)
    p.add_argument("--eps",            type=float, default=None)
    p.add_argument("--severity",       type=int, default=DEFAULT_SEVERITY)
    p.add_argument("--dataset",        default=None, choices=ALL_DATASETS)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--package",        action="store_true")
    p.add_argument("--upload_hf",      action="store_true")
    args = p.parse_args()

    if args.norm != "common" and args.eps is None:
        args.eps = {"Linf": 30, "L2": 2.0, "L1": 75}[args.norm]
        print(f"Using default eps={args.eps} for norm={args.norm}")

    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, OUTPUT_ROOT, PACKAGED_ROOT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ensure_data_downloaded(force=args.force_download)

    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    if args.norm == "common":
        for dataset in datasets:
            try:
                run_dataset_common(dataset, args)
            except Exception as e:
                print(f"\n  ERROR on {dataset}: {e}\n")
    else:
        device = get_device()
        for dataset in datasets:
            try:
                run_dataset(dataset, args, device)
            except Exception as e:
                print(f"\n  ERROR on {dataset}: {e}\n")

    print("\nAll done!")


if __name__ == "__main__":
    main()