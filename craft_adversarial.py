#!/usr/bin/env python3
"""
craft_adversarial.py — Craft L_inf adversarial perturbations using AutoAttack
against a zero-shot CLIP ViT-B/16 surrogate (open_clip, LAION-2B weights).

Loops over all datasets by default. Skips datasets already completed.
Writes metadata.jsonl incrementally after each batch (safe to kill and resume).

Usage:
    # All datasets
    python craft_adversarial.py --eps 30 --batch_size 8

    # Single dataset
    python craft_adversarial.py --dataset flowers-102 --eps 30 --batch_size 8

    # Quick sanity check
    python craft_adversarial.py --dataset flowers-102 --eps 30 --batch_size 4 --max_samples 32
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
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from autoattack import AutoAttack
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
# Data loading — copied exactly from llm_classify.py
# ---------------------------------------------------------------------------

def extract_archive(data_root: str, dataset_name: str, work_dir: str) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        print("pip install zstandard")
        sys.exit(1)

    archive_path = Path(data_root) / f"{dataset_name}_processed.tar.zst"
    dest_dir     = Path(work_dir)  / dataset_name

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if (dest_dir / "test" / "labels.csv").exists():
        return dest_dir

    print(f"Extracting {archive_path.name}...")
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


def load_class_names(class_names_dir: str, dataset_name: str) -> dict:
    path = Path(os.path.expanduser(class_names_dir)) / f"{dataset_name}.json"
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
# Zero-shot CLIP surrogate
# AutoAttack expects: model(x) → logits (B, N_classes), x in [0, 1]
# ---------------------------------------------------------------------------

CLIP_MODEL    = "ViT-B-16"
CLIP_PRETRAIN = "laion2b_s34b_b88k"
CLIP_MEAN     = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)


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


def load_zero_shot_clip(label_to_name: dict, device: torch.device) -> ZeroShotCLIP:
    print(f"\nLoading zero-shot CLIP: {CLIP_MODEL} / {CLIP_PRETRAIN}")
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAIN)
    clip_model.eval().to(device)

    tokenizer   = open_clip.get_tokenizer(CLIP_MODEL)
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    prompts     = [f"a photo of a {name}" for name in class_names]

    print(f"  Encoding {len(prompts)} class prompts...")
    with torch.no_grad():
        tokens        = tokenizer(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)

    print(f"  Text features: {text_features.shape}")
    model = ZeroShotCLIP(clip_model, text_features, device)
    model.eval().to(device)
    return model


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
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args, device: torch.device):
    eps_float = args.eps / 255.0
    run_name  = f"{dataset}__zeroshot_clip_vitb16_laion2b__linf_eps{args.eps}__autoattack_standard"
    output_dir = Path(os.path.expanduser(args.output_dir)) / run_name

    print(f"\n{'='*60}")
    print(f"  Dataset   : {dataset}")
    print(f"  eps       : {args.eps}/255 = {eps_float:.5f}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*60}")

    # Skip if already fully completed
    if (output_dir / "surrogate_summary.json").exists():
        print(f"  ✓ Already completed — skipping\n")
        return

    # Load data
    data_root     = os.path.expanduser(args.data_root)
    dataset_dir   = extract_archive(data_root, dataset, args.work_dir)
    items         = load_local_dataset(dataset_dir, split="test", max_samples=args.max_samples)
    label_to_name = load_class_names(args.class_names_dir, dataset)
    print(f"Loaded {len(items)} samples | {len(label_to_name)} classes")

    # Zero-shot CLIP surrogate (reloaded per dataset — class names differ)
    model = load_zero_shot_clip(label_to_name, device)

    # DataLoader
    transform = build_transform()
    ds        = AdversarialDataset(items, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "dataset":   dataset,
        "surrogate": f"zeroshot_{CLIP_MODEL}_{CLIP_PRETRAIN}",
        "attack":    "autoattack_standard",
        "eps_pixel": args.eps,
        "eps_float": eps_float,
        "norm":      "Linf",
    }, indent=2))

    # AutoAttack — full standard ensemble (APGD-CE + APGD-DLR + FAB + Square)
    adversary = AutoAttack(
        model, norm="Linf", eps=eps_float,
        version="standard",
        device=device, verbose=True,
    )

    n_correct_clean = 0
    n_correct_adv   = 0
    n_total         = 0

    # Open metadata.jsonl in append mode — safe to kill and partially resume
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

        # Save images + write metadata immediately
        records = save_batch(x_adv, labels_t.cpu().tolist(), filenames, label_to_name, output_dir)
        for rec in records:
            meta_file.write(json.dumps(rec) + "\n")
        meta_file.flush()

    meta_file.close()

    # Summary — written last so we can use its existence as completion flag
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
    p.add_argument("--dataset",         default=None,
                   help="Single dataset (omit to loop over all datasets)")
    p.add_argument("--eps",             type=int, default=4,
                   help="L_inf epsilon in pixel units [0-255], default=4")
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--data_root",       default="~/data_processed")
    p.add_argument("--class_names_dir", default="~/data_processed/class_names")
    p.add_argument("--work_dir",        default="/tmp/llm_classify")
    p.add_argument("--output_dir",      default="./adversarial_examples")
    p.add_argument("--max_samples",     type=int, default=None)
    args = p.parse_args()

    device   = get_device()
    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    print(f"\nProcessing {len(datasets)} dataset(s): {', '.join(datasets)}")

    for dataset in datasets:
        try:
            run_dataset(dataset, args, device)
        except Exception as e:
            print(f"\n  ERROR on {dataset}: {e} — skipping and continuing\n")
            continue

    print("\nAll done!")


if __name__ == "__main__":
    main()