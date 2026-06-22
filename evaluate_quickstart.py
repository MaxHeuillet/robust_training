#!/usr/bin/env python3
"""
evaluate_quickstart.py
======================
Minimal, self-contained script to evaluate ANY PyTorch model on RobustGenBench.

The model interface is intentionally plain:
    model(images: Tensor[B,3,H,W]) -> logits: Tensor[B, N]

Drop in any timm / torchvision / custom model — no CustomModel wrapper needed.

Evaluations run:
  • Clean test set
  • L-inf (eps=4/255), L2 (eps=2.0), L1 (eps=75.0) adversarial sets
    crafted for a specific surrogate model of your choice.

Dependencies:
    pip install torch torchvision huggingface_hub zstandard Pillow numpy

Usage:
    # Evaluate on flowers-102, attacks crafted on CLIP ViT-B/16
    python evaluate_quickstart.py \
        --dataset flowers-102 \
        --surrogate zeroshot_clip_vitb16_laion2b

    # List available surrogates / threat models and exit
    python evaluate_quickstart.py --list
"""

import argparse
import csv
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration — edit these to point at your own paths if needed
# ---------------------------------------------------------------------------

HF_REPO      = "legolasflagstaff/RobustGenBench"
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/tmp/hf_cache"))
WORK_DIR     = Path(os.environ.get("WORK_DIR",  "/tmp/robustgenbench_quickstart"))

# ---------------------------------------------------------------------------
# Available surrogates and threat models
# ---------------------------------------------------------------------------

SURROGATES = [
    "zeroshot_clip_vitb16_laion2b",
    "zeroshot_clip_vith14_laion2b",
    "zeroshot_metaclip_vith14_fullcc2_5b",
    "zeroshot_siglip2_so400m_patch14_384",
]

THREAT_MODELS = {
    "linf": "linf_eps4_autoattack_standard",
    "l2":   "l2_eps2_autoattack_standard",
    "l1":   "l1_eps75_autoattack_standard",
}

# ---------------------------------------------------------------------------
# Toy model — replace this with your own
# ---------------------------------------------------------------------------

class ToyModel(nn.Module):
    """
    A tiny CNN that works out-of-the-box with no downloads.
    Replace this class (or pass --model-path) with your real model.

    The only contract: forward(x) -> logits  (shape [B, num_classes])
    Normalization should be handled inside the model.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),  # 56x56
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 28x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                # 4x4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        # ImageNet-style normalization baked in
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


def load_model(num_classes: int, model_path: str = None) -> nn.Module:
    """
    Returns a model with the plain interface: model(x) -> logits.

    Swap out ToyModel for your own class here, or load weights from
    model_path if provided.

    Examples:
        # timm
        import timm
        model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)

        # torchvision
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    """
    model = ToyModel(num_classes)
    if model_path:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from {model_path}")
    return model

# ---------------------------------------------------------------------------
# HuggingFace helpers  (mirrors evaluate_robustgenbench.py)
# ---------------------------------------------------------------------------

def get_num_classes(dataset: str) -> int:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"class_names/{dataset}.json",
        local_dir=str(WORK_DIR / "class_names"),
        cache_dir=str(HF_CACHE_DIR),
    )
    with open(path) as f:
        return len(json.load(f))


def find_archive_name(hf_dir: str, dataset: str) -> str | None:
    from huggingface_hub import list_repo_tree
    try:
        for entry in list_repo_tree(repo_id=HF_REPO, repo_type="dataset",
                                    path_in_repo=hf_dir):
            name = Path(entry.path).name
            if name.startswith(dataset) and name.endswith(".tar.zst"):
                return name
    except Exception as e:
        print(f"  WARNING: could not list {hf_dir}: {e}")
    return None


def download_archive(hf_dir: str, archive_name: str, local_dir: Path) -> Path | None:
    from huggingface_hub import hf_hub_download
    local_path = local_dir / archive_name
    if local_path.exists():
        print(f"  Already cached: {archive_name}")
        return local_path
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {hf_dir}/{archive_name}")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=f"{hf_dir}/{archive_name}",
            local_dir=str(local_dir),
            cache_dir=str(HF_CACHE_DIR),
        )
        src = Path(downloaded)
        if src != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            src.rename(local_path)
        return local_path
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None


def extract_archive(archive_path: Path, extract_root: Path) -> Path | None:
    try:
        import zstandard as zstd
    except ImportError:
        print("ERROR: zstandard not installed. Run: pip install zstandard")
        sys.exit(1)

    stem    = archive_path.name.replace("_processed.tar.zst", "")
    dest    = extract_root / stem
    sentinel = dest / "test" / "labels.csv"

    if sentinel.exists():
        print(f"  Already extracted: {dest.name}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} ...")
    try:
        with open(archive_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=dest)
        return dest
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return None


def read_labels_csv(extract_dir: Path) -> list[tuple[Path, int]]:
    csv_path = extract_dir / "test" / "labels.csv"
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append((extract_dir / "test" / row["filename"], int(row["label"])))
    return items

# ---------------------------------------------------------------------------
# Dataset + transform
# ---------------------------------------------------------------------------

class BenchmarkDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def build_transform():
    # Images are stored as 224×224; we load as float tensors in [0, 1].
    # Your model is responsible for normalization.
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),   # -> [0, 1]
    ])

# ---------------------------------------------------------------------------
# Evaluation loop  (single GPU / CPU, no DDP — keep it simple)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, items: list, device: torch.device,
             batch_size: int = 64) -> dict:
    transform = build_transform()
    loader    = DataLoader(
        BenchmarkDataset(items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    nb_correct  = 0
    nb_examples = 0
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)          # plain interface: model(x) -> logits
        preds  = logits.argmax(dim=1)
        nb_correct  += (preds == labels).sum().item()
        nb_examples += labels.size(0)

    accuracy = nb_correct / nb_examples if nb_examples > 0 else 0.0
    return {"nb_correct": nb_correct, "nb_examples": nb_examples,
            "accuracy": round(accuracy, 4)}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_clean_items(dataset: str) -> list | None:
    """Download and extract the clean test split."""
    from huggingface_hub import hf_hub_download
    archive_name    = f"{dataset}_processed.tar.zst"
    archives_dir    = WORK_DIR / "clean_archives"
    extract_root    = WORK_DIR / "clean"
    archives_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archives_dir / archive_name
    if not archive_path.exists():
        print(f"  Downloading clean archive: {archive_name}")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset",
                filename=archive_name,
                local_dir=str(archives_dir),
                cache_dir=str(HF_CACHE_DIR),
            )
            src = Path(downloaded)
            if src != archive_path:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                src.rename(archive_path)
        except Exception as e:
            print(f"  WARNING: could not download clean archive: {e}")
            return None

    extract_dir = extract_archive(archive_path, extract_root)
    if extract_dir is None:
        return None
    return read_labels_csv(extract_dir)


def load_adv_items(dataset: str, surrogate: str, threat_key: str) -> list | None:
    """Download and extract one adversarial test split."""
    threat_name = THREAT_MODELS[threat_key]
    hf_dir      = f"adversarial/{surrogate}/{threat_name}"
    archive_name = find_archive_name(hf_dir, dataset)
    if archive_name is None:
        print(f"  WARNING: no archive found in {hf_dir} for dataset '{dataset}'")
        return None

    archives_dir  = WORK_DIR / "adv_archives" / dataset
    extract_root  = WORK_DIR / "adv_extracted" / dataset / surrogate / threat_name
    archive_path  = download_archive(hf_dir, archive_name, archives_dir)
    if archive_path is None:
        return None

    extract_dir = extract_archive(archive_path, extract_root)
    if extract_dir is None:
        return None

    return read_labels_csv(extract_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PyTorch model on RobustGenBench."
    )
    parser.add_argument("--dataset",    default="flowers-102",
                        help="Dataset name on RobustGenBench (e.g. flowers-102)")
    parser.add_argument("--surrogate",  default="zeroshot_clip_vitb16_laion2b",
                        choices=SURROGATES,
                        help="Surrogate model used to craft adversarial examples")
    parser.add_argument("--threat",     nargs="+", default=["linf", "l2", "l1"],
                        choices=list(THREAT_MODELS.keys()),
                        help="Which threat models to evaluate (default: all three)")
    parser.add_argument("--model-path", default=None,
                        help="Optional path to a .pt state dict to load into ToyModel")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--results-dir", default="./quickstart_results",
                        help="Directory to write the output CSV")
    parser.add_argument("--list",       action="store_true",
                        help="Print available surrogates and threat models, then exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable surrogates:")
        for s in SURROGATES:
            print(f"  {s}")
        print("\nAvailable threat models:")
        for k, v in THREAT_MODELS.items():
            print(f"  --threat {k}  →  {v}")
        sys.exit(0)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n  Device: {device}")

    # ---- num_classes -------------------------------------------------------
    print(f"  Fetching class list for '{args.dataset}' ...")
    N = get_num_classes(args.dataset)
    print(f"  num_classes = {N}")

    # ---- model -------------------------------------------------------------
    model = load_model(N, args.model_path).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model.__class__.__name__}  ({total_params:,} parameters)")
    print()
    print("  NOTE: ToyModel is randomly initialised — accuracy will be near chance.")
    print("  Replace load_model() with your own model to get real numbers.")

    # ---- output CSV --------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = results_dir / f"{args.dataset}__{args.surrogate}__results.csv"

    fieldnames  = ["split", "surrogate", "threat_model",
                   "nb_correct", "nb_examples", "accuracy"]
    rows        = []

    # ---- clean eval --------------------------------------------------------
    print(f"{'='*60}")
    print("  Split: CLEAN")
    clean_items = load_clean_items(args.dataset)
    if clean_items:
        print(f"  {len(clean_items)} images")
        # Pixel sanity check
        arr = np.array(Image.open(clean_items[0][0]).convert("RGB")) / 255.0
        print(f"  Pixel range — min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
        stats = evaluate(model, clean_items, device, args.batch_size)
        print(f"  Accuracy: {stats['nb_correct']}/{stats['nb_examples']} = {stats['accuracy']:.4f}")
        rows.append({"split": "clean", "surrogate": "none",
                     "threat_model": "clean", **stats})
    else:
        print("  WARNING: clean items could not be loaded, skipping.")

    # ---- adversarial evals -------------------------------------------------
    for threat_key in args.threat:
        threat_name = THREAT_MODELS[threat_key]
        print(f"\n{'='*60}")
        print(f"  Split: {threat_key.upper()}  |  surrogate: {args.surrogate}")
        print(f"  Threat model: {threat_name}")

        adv_items = load_adv_items(args.dataset, args.surrogate, threat_key)
        if adv_items is None:
            print("  WARNING: could not load adversarial data, skipping.")
            continue

        print(f"  {len(adv_items)} images")
        arr = np.array(Image.open(adv_items[0][0]).convert("RGB")) / 255.0
        print(f"  Pixel range — min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")

        stats = evaluate(model, adv_items, device, args.batch_size)
        print(f"  Accuracy: {stats['nb_correct']}/{stats['nb_examples']} = {stats['accuracy']:.4f}")
        rows.append({"split": threat_key, "surrogate": args.surrogate,
                     "threat_model": threat_name, **stats})

    # ---- write CSV ---------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {csv_path}")

    # ---- summary table -----------------------------------------------------
    print(f"\n  {'Split':<12} {'Accuracy':>10}  nb_correct / nb_examples")
    print(f"  {'-'*50}")
    for row in rows:
        print(f"  {row['split']:<12} {row['accuracy']:>10.4f}  "
              f"{row['nb_correct']}/{row['nb_examples']}")
    print()


if __name__ == "__main__":
    main()