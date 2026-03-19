#!/usr/bin/env python3
"""
craft_adversarial.py — Craft L_inf adversarial perturbations using AutoAttack
against either:
  • zero-shot CLIP ViT-B/16   (open_clip, LAION-2B weights)
  • zero-shot SigLIP2-base-patch16-224  (HuggingFace transformers)

Data is downloaded automatically from HuggingFace Hub (MaxHeuillet/RobustGenBench)
on first run. Everything is stored under /tmp to stay cluster-friendly.

Usage:
    # CLIP surrogate (default), all datasets
    python craft_adversarial.py --surrogate clip --eps 30 --batch_size 8

    # SigLIP2 surrogate, all datasets
    python craft_adversarial.py --surrogate siglip2 --eps 30 --batch_size 8

    # Single dataset
    python craft_adversarial.py --surrogate siglip2 --dataset flowers-102 --eps 30 --batch_size 8

    # Quick sanity check
    python craft_adversarial.py --surrogate siglip2 --dataset flowers-102 --eps 30 --batch_size 4 --max_samples 32

    # Force re-download even if data already present
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
# Cluster-friendly /tmp paths (no home-dir quota issues)
# ---------------------------------------------------------------------------

TMP_ROOT       = Path("/tmp/robustgenbench")
DATA_ROOT      = TMP_ROOT / "data_processed"       # extracted archives land here
HF_CACHE_DIR   = TMP_ROOT / "hf_cache"             # HuggingFace model + dataset cache
OUTPUT_ROOT    = TMP_ROOT / "adversarial_examples"  # adversarial images output
WORK_DIR       = TMP_ROOT / "work"                  # temp extraction workspace

HF_DATASET_REPO = "MaxHeuillet/RobustGenBench"
CLASS_NAMES_DIR = DATA_ROOT / "class_names"


# ---------------------------------------------------------------------------
# Surrogate identifiers
# ---------------------------------------------------------------------------

SURROGATE_CLIP   = "clip"
SURROGATE_SIGLIP = "siglip2"
ALL_SURROGATES   = [SURROGATE_CLIP, SURROGATE_SIGLIP]

# CLIP config
CLIP_MODEL      = "ViT-B-16"
CLIP_PRETRAIN   = "laion2b_s34b_b88k"
CLIP_MEAN       = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD        = (0.26862954, 0.26130258, 0.27577711)

# SigLIP2 config — base patch16 224 is natively 224×224, no resize needed
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_MEAN     = (0.5, 0.5, 0.5)
SIGLIP_STD      = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]


# ---------------------------------------------------------------------------
# Device: CUDA > MPS (Apple M1) > CPU
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
# Automatic data download
# ---------------------------------------------------------------------------

def ensure_data_downloaded(force: bool = False):
    """
    Downloads the full RobustGenBench dataset from HuggingFace Hub into
    DATA_ROOT (/tmp/robustgenbench/data_processed) if not already present.

    Sets HF_HOME to HF_CACHE_DIR so all HuggingFace artefacts stay in /tmp.
    """
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

    # Sentinel file avoids re-downloading on every run
    sentinel = DATA_ROOT / ".download_complete"
    if sentinel.exists() and not force:
        print(f"Data already present at {DATA_ROOT} (use --force_download to re-fetch).")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    print(f"\nDownloading dataset {HF_DATASET_REPO!r} → {DATA_ROOT}")
    print("This may take a while on first run...\n")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
        cache_dir=str(HF_CACHE_DIR),
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
        raise FileNotFoundError(
            f"Archive not found: {archive_path}\n"
            f"Run with --force_download to re-fetch the dataset."
        )

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
    csv_path  = split_dir  / "labels.csv"
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
# AutoAttack expects: model(x) → logits (B, N_classes), x in [0, 1]
# ---------------------------------------------------------------------------

class ZeroShotCLIP(nn.Module):
    """
    CLIP ViT-B/16 zero-shot surrogate (open_clip, LAION-2B weights).
    Normalisation is applied inside forward so AutoAttack sees raw [0,1] input.
    """

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
    """
    SigLIP2-base-patch16-224 zero-shot surrogate (HuggingFace transformers).

    SigLIP uses a sigmoid-based contrastive loss rather than softmax, so its
    raw similarity scores are not calibrated as probabilities. We follow the
    standard zero-shot recipe: cosine similarity × temperature, then let
    AutoAttack treat the result as logits.

    Normalisation (mean=0.5, std=0.5) is applied inside forward so
    AutoAttack sees raw [0,1] input — same contract as ZeroShotCLIP.

    IMPORTANT — opaque inner module:
    AutoAttack (≥0.3) serialises the model to inspect its architecture using a
    Rust-backed JSON parser. HuggingFace PreTrainedModel configs contain fields
    (e.g. model_type-specific union variants) that trip this parser with:
        "data did not match any variant of untagged enum ModelWrapper"
    The fix is to store the HuggingFace vision model inside a plain Python
    callable (_SigLIPEncoder) that is NOT registered as an nn.Module submodule.
    AutoAttack only walks nn.Module children — anything stored as a plain
    attribute is invisible to its introspection, so the problematic config
    never gets serialised.
    """

    def __init__(self, encode_fn, text_features: torch.Tensor, device, temperature: float = 100.0):
        super().__init__()
        # encode_fn is a plain Python callable, NOT an nn.Module child.
        # This keeps the HuggingFace PreTrainedModel out of AutoAttack's
        # module-tree serialiser, avoiding the ModelWrapper JSON parse error.
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
# Dataset-specific prompt templates for zero-shot classification.
# Generic "a photo of a {name}" works for coarse-grained datasets but fails
# for fine-grained ones where class names are highly specific (car models,
# aircraft variants, pet breeds). Templates are taken from the CLIP/SigLIP
# zero-shot evaluation literature.
# ---------------------------------------------------------------------------

DATASET_PROMPT_TEMPLATES = {
    "caltech101":                lambda name: f"a photo of a {name}",
    "fgvc-aircraft-2013b":       lambda name: f"a photo of a {name}, a type of aircraft",
    "flowers-102":               lambda name: f"a photo of a {name}, a type of flower",
    "oxford-iiit-pet":           lambda name: f"a photo of a {name}, a type of pet",
    "stanford_cars":             lambda name: f"a photo of a {name}",
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
        CLIP_MODEL, pretrained=CLIP_PRETRAIN,
        cache_dir=str(HF_CACHE_DIR),
    )
    clip_model.eval().to(device)

    tokenizer   = open_clip.get_tokenizer(CLIP_MODEL)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)

    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        tokens        = tokenizer(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)

    print(f"  Text features: {text_features.shape}")
    model = ZeroShotCLIP(clip_model, text_features, device)
    model.eval().to(device)
    return model


def load_siglip2_surrogate(label_to_name: dict, device: torch.device, dataset: str = "") -> ZeroShotSigLIP2:
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel

    print(f"\nLoading SigLIP2 surrogate: {SIGLIP_MODEL_ID}")

    vision_model = SiglipVisionModel.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR)
    )
    vision_model.eval().to(device)

    text_model = SiglipTextModel.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR)
    )
    text_model.eval().to(device)

    # use_fast=False avoids the Rust tokenizer which fails to parse the
    # Gemma-based SigLIP2 tokenizer.json on older `tokenizers` versions.
    tokenizer   = AutoTokenizer.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=str(HF_CACHE_DIR), use_fast=False
    )
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = build_prompts(dataset, class_names)

    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        # padding="max_length" with the model's own max_position_embeddings is
        # required for SigLIP2: pooler_output pools at the EOS token position,
        # which is only correct when every sequence is padded to the fixed
        # length the model was trained with.
        max_len = text_model.config.max_position_embeddings
        inputs        = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        text_outputs  = text_model(**inputs)
        text_features = F.normalize(text_outputs.pooler_output, dim=-1)

    print(f"  Text features: {text_features.shape}")

    # Free text model before the attack loop to save GPU memory
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Wrap vision_model in a plain Python callable (NOT an nn.Module).
    # This keeps HuggingFace's PreTrainedModel out of AutoAttack's module-tree
    # serialiser, which chokes on SigLIP2's config with:
    #   "data did not match any variant of untagged enum ModelWrapper"
    # A lambda/closure is not an nn.Module child, so AutoAttack never sees it.
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
    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}. Choose from {ALL_SURROGATES}")


def surrogate_slug(surrogate: str) -> str:
    if surrogate == SURROGATE_CLIP:
        return "zeroshot_clip_vitb16_laion2b"
    elif surrogate == SURROGATE_SIGLIP:
        return "zeroshot_siglip2_base_patch16_224"
    raise ValueError(surrogate)


def build_transform() -> T.Compose:
    """Both surrogates operate at 224×224 — single shared transform."""
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
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args, device: torch.device):
    eps_float  = args.eps / 255.0
    slug       = surrogate_slug(args.surrogate)
    run_name   = f"{dataset}__{slug}__linf_eps{args.eps}__autoattack_standard"
    output_dir = OUTPUT_ROOT / run_name

    print(f"\n{'='*60}")
    print(f"  Dataset   : {dataset}")
    print(f"  Surrogate : {args.surrogate}")
    print(f"  eps       : {args.eps}/255 = {eps_float:.5f}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*60}")

    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping\n")
        return

    dataset_dir   = extract_archive(dataset)
    items         = load_local_dataset(dataset_dir, split="test", max_samples=args.max_samples)
    label_to_name = load_class_names(dataset)
    print(f"Loaded {len(items)} samples | {len(label_to_name)} classes")

    # Surrogate reloaded per dataset — class text features differ per dataset
    model = load_surrogate(args.surrogate, label_to_name, device, dataset=dataset)

    transform = build_transform()
    ds        = AdversarialDataset(items, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset":        dataset,
        "surrogate":      args.surrogate,
        "surrogate_slug": slug,
        "attack":         "autoattack_standard",
        "eps_pixel":      args.eps,
        "eps_float":      eps_float,
        "norm":           "Linf",
    }, indent=2))

    adversary = AutoAttack(
        model, norm="Linf", eps=eps_float,
        version="standard",
        device=device, verbose=True,
    )

    n_correct_clean = 0
    n_correct_adv   = 0
    n_total         = 0

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

    print(f"\n{'='*60}")
    print(f"  {n_total} images saved to : {output_dir}")
    print(f"  Surrogate clean acc  : {clean_acc:.4f}")
    print(f"  Surrogate adv acc    : {adv_acc:.4f}")
    print(f"  Attack success rate  : {1 - adv_acc:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate",      default=SURROGATE_CLIP,
                   choices=ALL_SURROGATES,
                   help=f"Surrogate model. One of: {ALL_SURROGATES} (default: clip)")
    p.add_argument("--dataset",        default=None,
                   help="Single dataset name (omit to loop over all datasets)")
    p.add_argument("--eps",            type=int, default=4,
                   help="L_inf epsilon in pixel units [0-255], default=4")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--force_download", action="store_true",
                   help="Re-download the dataset even if already present in /tmp")
    args = p.parse_args()

    # Ensure all /tmp dirs exist upfront
    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, OUTPUT_ROOT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Download data if needed (no-op if sentinel present and not forced)
    ensure_data_downloaded(force=args.force_download)

    device   = get_device()
    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    print(f"\nSurrogate : {args.surrogate}")
    print(f"Data      : {DATA_ROOT}")
    print(f"Output    : {OUTPUT_ROOT}")
    print(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")

    for dataset in datasets:
        try:
            run_dataset(dataset, args, device)
        except Exception as e:
            print(f"\n  ERROR on {dataset}: {e} — skipping and continuing\n")
            continue

    print("\nAll done!")


if __name__ == "__main__":
    main()