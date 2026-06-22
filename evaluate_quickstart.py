#!/usr/bin/env python3
"""
evaluate_quickstart.py
======================
White-box adversarial evaluation of ANY PyTorch model on RobustGenBench.

Attacks are crafted on-the-fly with AutoAttack directly against the user's
model (white-box), matching the evaluation protocol of the benchmark.

Model interface — intentionally plain:
    model(images: Tensor[B,3,H,W]) -> logits: Tensor[B, N]

A thin WhiteBoxWrapper adapts this to the internal (x_nat, x_adv) interface
used by the eval loop, so no CustomModel boilerplate is needed.

Evaluations:
  • Clean accuracy   — forward pass on the original test images
  • L-inf (eps=4/255), L2 (eps=2.0), L1 (eps=75.0) — AutoAttack white-box

Dependencies:
    pip install torch torchvision huggingface_hub zstandard Pillow numpy
    pip install git+https://github.com/fra31/auto-attack

Usage:
    # Single GPU (default)
    python evaluate_quickstart.py --dataset flowers-102

    # Multi-GPU
    python evaluate_quickstart.py --dataset flowers-102 --multi-gpu

    # Subset of threat models
    python evaluate_quickstart.py --dataset flowers-102 --threat linf l2

    # Print training data location + DataLoader snippet after eval
    python evaluate_quickstart.py --dataset flowers-102 --show-train-data

    # List available threat models and exit
    python evaluate_quickstart.py --list
"""

import argparse
import csv
import json
import os
import sys
import tarfile
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO      = "legolasflagstaff/RobustGenBench"
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/tmp/hf_cache"))
WORK_DIR     = Path(os.environ.get("WORK_DIR",  "/tmp/robustgenbench_quickstart"))

# Epsilon values matching the benchmark's AutoAttack standard protocol
THREAT_MODELS = {
    "linf": {"norm": "Linf", "eps": 4  / 255},
    "l2":   {"norm": "L2",   "eps": 2.0},
    "l1":   {"norm": "L1",   "eps": 75.0},
}

# ---------------------------------------------------------------------------
# Toy model — replace with your own
# ---------------------------------------------------------------------------

class ToyModel(nn.Module):
    """
    A tiny CNN that works out-of-the-box with no downloads.
    Replace this class (or the body of load_model()) with your real model.

    The only contract:  forward(x) -> logits  (shape [B, num_classes])
    Normalization must be handled inside the model (inputs are in [0, 1]).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),   # -> 56x56
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # -> 28x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                 # -> 4x4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        # ImageNet-style normalization baked in so inputs can stay in [0, 1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


def load_model(num_classes: int, model_path: str = None) -> nn.Module:
    """
    Returns a model with the plain interface: model(x) -> logits.

    ── Swap examples ───────────────────────────────────────────────────────
    # timm
    import timm
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)

    # torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ────────────────────────────────────────────────────────────────────────
    """
    model = ToyModel(num_classes)
    if model_path:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from {model_path}")
    return model

# ---------------------------------------------------------------------------
# WhiteBoxWrapper
# ---------------------------------------------------------------------------

class WhiteBoxWrapper(nn.Module):
    """
    Adapts a plain model(x) -> logits to the two-input interface used by the
    eval loop and by AutoAttack:

        wrapper(x)            -> logits          (used by AutoAttack)
        wrapper(x_nat, x_adv) -> (logits_nat, logits_adv)  (used by eval loop)

    This is the only glue needed — users never touch this class.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_nat: torch.Tensor,
                x_adv: torch.Tensor = None):
        logits_nat = self.model(x_nat)
        if x_adv is None:
            # AutoAttack calls wrapper(x) for gradient-based attack
            return logits_nat
        logits_adv = self.model(x_adv)
        return logits_nat, logits_adv

# ---------------------------------------------------------------------------
# Dataset + transform
# ---------------------------------------------------------------------------

class BenchmarkDataset(Dataset):
    def __init__(self, items: list, transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def build_transform():
    # Images are stored as 224×224 JPEGs.
    # ToTensor() maps [0,255] -> [0,1]; normalization is inside the model.
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

# ---------------------------------------------------------------------------
# Eval loops  (mirrors test_loop() from distributed_experiment_final.py)
# ---------------------------------------------------------------------------

def eval_clean(model: WhiteBoxWrapper, loader: DataLoader,
               device: torch.device) -> tuple[int, int]:
    """Forward pass on clean images — no attack."""
    nb_correct, nb_examples = 0, 0
    model.eval()
    with torch.no_grad():
        for x_nat, target in loader:
            x_nat, target = x_nat.to(device), target.to(device)
            logits_nat, _ = model(x_nat, x_nat)   # x_adv = x_nat (no attack)
            preds = logits_nat.argmax(dim=1)
            nb_correct  += (preds == target).sum().item()
            nb_examples += target.size(0)
    return nb_correct, nb_examples


def eval_adversarial(model: WhiteBoxWrapper, loader: DataLoader,
                     device: torch.device, norm: str,
                     eps: float, batch_size: int) -> tuple[int, int, int]:
    """
    Mirrors the Linf / L2 / L1 branch of test_loop() in
    distributed_experiment_final.py.

    For each batch:
      1. WhiteBoxWrapper(x) -> logits is passed to AutoAttack as the
         forward callable (single-argument form), matching the
         `forward_pass = lambda x: model(x)` pattern in test_loop().
      2. AutoAttack crafts x_adv via run_standard_evaluation(), same call
         as in test_loop().
      3. Both x_nat and x_adv are forwarded in one call to get
         (logits_nat, logits_adv), matching
         `logits_nat, logits_adv = model(x_nat, x_adv)` in test_loop().
      4. Top-1 accuracy is accumulated for both, same as test_loop().
    """
    from autoattack import AutoAttack

    nb_correct_nat, nb_correct_adv, nb_examples = 0, 0, 0
    model.eval()

    # WhiteBoxWrapper satisfies AutoAttack's expected callable model(x) -> logits
    # when called with a single argument (x_adv=None path).
    adversary = AutoAttack(model, norm=norm, eps=eps,
                           version="standard", verbose=False, device=device)

    for x_nat, target in loader:
        x_nat, target = x_nat.to(device), target.to(device)

        x_adv = adversary.run_standard_evaluation(x_nat, target, bs=batch_size)

        with torch.no_grad():
            logits_nat, logits_adv = model(x_nat, x_adv)

        nb_correct_nat += (logits_nat.argmax(1) == target).sum().item()
        nb_correct_adv += (logits_adv.argmax(1) == target).sum().item()
        nb_examples    += target.size(0)

    return nb_correct_nat, nb_correct_adv, nb_examples


def _worker(rank: int, world_size: int, items: list, model_state: dict,
            num_classes: int, model_path: str, threat_key: str,
            batch_size: int, result_queue: Queue) -> None:
    """
    Worker for multi-GPU evaluation via mp.Process (mirrors launch_test /
    test() in distributed_experiment_final.py).
    Each rank handles its own shard via DistributedSampler.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    plain_model = load_model(num_classes, model_path)
    plain_model.load_state_dict(model_state)
    model = WhiteBoxWrapper(plain_model).to(device)
    model.eval()

    transform = build_transform()
    dataset   = BenchmarkDataset(items, transform)
    sampler   = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                   shuffle=False, drop_last=False)
    loader    = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                           num_workers=2, pin_memory=True)

    if threat_key == "clean":
        nb_correct, nb_examples = eval_clean(model, loader, device)
        result_queue.put((rank, nb_correct, None, nb_examples))
    else:
        cfg = THREAT_MODELS[threat_key]
        nb_correct_nat, nb_correct_adv, nb_examples = eval_adversarial(
            model, loader, device, cfg["norm"], cfg["eps"], batch_size)
        result_queue.put((rank, nb_correct_nat, nb_correct_adv, nb_examples))

    dist.barrier()
    dist.destroy_process_group()


def run_multi_gpu(items: list, model_state: dict, num_classes: int,
                  model_path: str, threat_key: str,
                  batch_size: int) -> dict:
    """Spawn one process per GPU, aggregate results."""
    world_size   = torch.cuda.device_count()
    result_queue = Queue()
    mp.spawn(_worker,
             args=(world_size, items, model_state, num_classes,
                   model_path, threat_key, batch_size, result_queue),
             nprocs=world_size, join=True)

    total_nat, total_adv, total_examples = 0, 0, 0
    while not result_queue.empty():
        rank, nb_nat, nb_adv, nb_ex = result_queue.get()
        total_nat     += nb_nat
        total_adv     += nb_adv if nb_adv is not None else 0
        total_examples += nb_ex

    if threat_key == "clean":
        acc = total_nat / total_examples if total_examples else 0.0
        return {"nb_correct": total_nat, "nb_correct_adv": None,
                "nb_examples": total_examples, "accuracy_nat": round(acc, 4)}
    else:
        acc_nat = total_nat / total_examples if total_examples else 0.0
        acc_adv = total_adv / total_examples if total_examples else 0.0
        return {"nb_correct_nat": total_nat, "nb_correct_adv": total_adv,
                "nb_examples": total_examples,
                "accuracy_nat": round(acc_nat, 4),
                "accuracy_adv": round(acc_adv, 4)}


def run_single_gpu(items: list, model: WhiteBoxWrapper,
                   device: torch.device, threat_key: str,
                   batch_size: int) -> dict:
    """Single-device evaluation — simpler, no process spawning."""
    transform = build_transform()
    loader    = DataLoader(BenchmarkDataset(items, transform),
                           batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=device.type == "cuda")

    if threat_key == "clean":
        nb_correct, nb_examples = eval_clean(model, loader, device)
        acc = nb_correct / nb_examples if nb_examples else 0.0
        return {"nb_correct": nb_correct, "nb_correct_adv": None,
                "nb_examples": nb_examples, "accuracy_nat": round(acc, 4)}
    else:
        cfg = THREAT_MODELS[threat_key]
        nb_nat, nb_adv, nb_examples = eval_adversarial(
            model, loader, device, cfg["norm"], cfg["eps"], batch_size)
        acc_nat = nb_nat / nb_examples if nb_examples else 0.0
        acc_adv = nb_adv / nb_examples if nb_examples else 0.0
        return {"nb_correct_nat": nb_nat, "nb_correct_adv": nb_adv,
                "nb_examples": nb_examples,
                "accuracy_nat": round(acc_nat, 4),
                "accuracy_adv": round(acc_adv, 4)}

# ---------------------------------------------------------------------------
# HuggingFace helpers
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


def download_archive(hf_dir: str, archive_name: str,
                     local_dir: Path) -> Path | None:
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
        print("ERROR: zstandard not installed.  pip install zstandard")
        sys.exit(1)

    stem     = archive_path.name.replace("_processed.tar.zst", "")
    dest     = extract_root / stem
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


def read_labels_csv(extract_dir: Path, split: str = "test") -> list:
    csv_path = extract_dir / split / "labels.csv"
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append((extract_dir / split / row["filename"], int(row["label"])))
    return items


def load_clean_items(dataset: str) -> list | None:
    from huggingface_hub import hf_hub_download
    archive_name = f"{dataset}_processed.tar.zst"
    archives_dir = WORK_DIR / "clean_archives"
    extract_root = WORK_DIR / "clean"
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
    return read_labels_csv(extract_dir, split="test") if extract_dir else None

# ---------------------------------------------------------------------------
# Training data info
# ---------------------------------------------------------------------------

def show_training_data_info(dataset: str) -> None:
    """
    Print the layout of the extracted clean archive and a ready-to-use
    DataLoader snippet so users can train before evaluating.

    The archive contains three splits:  train / val / test
    Each split folder has:
        labels.csv   — columns: filename, label  (0-based integer class index)
        <images>     — JPEG files already resized to 224×224
    """
    extract_root = WORK_DIR / "clean"
    candidates   = list(extract_root.glob(f"{dataset}*"))
    if not candidates:
        print("  Training data not found locally — run the clean evaluation "
              "first so the archive is downloaded and extracted.")
        return

    dataset_dir = candidates[0]
    print(f"\n{'='*60}")
    print(f"  Training data location")
    print(f"{'='*60}")
    print(f"  Root : {dataset_dir}")
    print()

    for split in ("train", "val", "test"):
        csv_path = dataset_dir / split / "labels.csv"
        if not csv_path.exists():
            print(f"  {split:<6}  — not found")
            continue
        with open(csv_path) as f:
            n = sum(1 for _ in csv.DictReader(f))
        print(f"  {split:<6}  {n:>6} images   →  {dataset_dir / split}")

    print()
    print("  Image format : JPEG, 224×224, pixel values in [0, 255]")
    print("  Label format : integer class index (0-based), see labels.csv")
    print()

    class_names_path = WORK_DIR / "class_names" / f"{dataset}.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            raw = json.load(f)
        label_to_name = ({int(k): v for k, v in raw.items()}
                         if isinstance(raw, dict)
                         else {i: v for i, v in enumerate(raw)})
        print("  Class index sample:")
        for idx in sorted(label_to_name)[:5]:
            print(f"    {idx:>4} → {label_to_name[idx]}")
        if len(label_to_name) > 5:
            print(f"    ... ({len(label_to_name)} classes total)")
        print()

    print("  ── How to build a training DataLoader ─────────────────────────")
    print(f"""
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    train_items = read_labels_csv(Path("{dataset_dir}"), split="train")
    val_items   = read_labels_csv(Path("{dataset_dir}"), split="val")

    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),          # -> [0, 1]; normalise inside your model
    ])
    val_tf = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    train_loader = DataLoader(
        BenchmarkDataset(train_items, train_tf),
        batch_size=64, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        BenchmarkDataset(val_items, val_tf),
        batch_size=64, shuffle=False, num_workers=4,
    )
    # Then: for images, labels in train_loader: ...
""")
    print("  ────────────────────────────────────────────────────────────────")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="White-box adversarial evaluation on RobustGenBench."
    )
    parser.add_argument("--dataset",    default="flowers-102",
                        help="Dataset name on RobustGenBench (e.g. flowers-102)")
    parser.add_argument("--threat",     nargs="+",
                        default=["linf", "l2", "l1"],
                        choices=list(THREAT_MODELS.keys()),
                        help="Threat models to evaluate (default: all three)")
    parser.add_argument("--model-path", default=None,
                        help="Optional path to a .pt state dict for ToyModel")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--multi-gpu",  action="store_true",
                        help="Distribute evaluation across all available GPUs")
    parser.add_argument("--results-dir", default="./quickstart_results")
    parser.add_argument("--show-train-data", action="store_true",
                        help="Print training data location + DataLoader snippet")
    parser.add_argument("--list",       action="store_true",
                        help="Print threat model options and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable threat models:")
        for k, v in THREAT_MODELS.items():
            print(f"  --threat {k:<6}  norm={v['norm']}  eps={v['eps']}")
        sys.exit(0)

    # ---- device ------------------------------------------------------------
    if args.multi_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("WARNING: --multi-gpu requested but fewer than 2 GPUs found. "
                  "Falling back to single GPU.")
            args.multi_gpu = False

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_gpus = torch.cuda.device_count() if args.multi_gpu else 1
    print(f"\n  Device     : {device}")
    print(f"  Multi-GPU  : {args.multi_gpu}  ({n_gpus} GPU(s))")

    # ---- num_classes -------------------------------------------------------
    print(f"  Fetching class list for '{args.dataset}' ...")
    N = get_num_classes(args.dataset)
    print(f"  num_classes = {N}")

    # ---- model -------------------------------------------------------------
    plain_model = load_model(N, args.model_path)
    model       = WhiteBoxWrapper(plain_model).to(device)
    total_params = sum(p.numel() for p in plain_model.parameters())
    print(f"  Model      : {plain_model.__class__.__name__}  ({total_params:,} params)")
    print()
    print("  NOTE: ToyModel is randomly initialised — accuracy will be near chance.")
    print("  Replace load_model() with your own model to get real numbers.")
    print()

    # For multi-GPU we pass the state dict to each worker
    model_state = plain_model.state_dict() if args.multi_gpu else None

    # ---- output CSV --------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = results_dir / f"{args.dataset}__whitebox__results.csv"
    fieldnames  = ["split", "norm", "eps",
                   "nb_correct_nat", "nb_correct_adv", "nb_examples",
                   "accuracy_nat", "accuracy_adv"]
    rows = []

    # ---- clean eval --------------------------------------------------------
    print(f"{'='*60}")
    print("  Split: CLEAN")
    clean_items = load_clean_items(args.dataset)
    if clean_items:
        print(f"  {len(clean_items)} images")
        arr = np.array(Image.open(clean_items[0][0]).convert("RGB")) / 255.0
        print(f"  Pixel range — min={arr.min():.3f}  max={arr.max():.3f}  "
              f"mean={arr.mean():.3f}")
        if args.multi_gpu:
            stats = run_multi_gpu(clean_items, model_state, N,
                                  args.model_path, "clean", args.batch_size)
        else:
            stats = run_single_gpu(clean_items, model, device,
                                   "clean", args.batch_size)
        print(f"  Clean accuracy: "
              f"{stats['nb_correct']}/{stats['nb_examples']} "
              f"= {stats['accuracy_nat']:.4f}")
        rows.append({"split": "clean", "norm": "—", "eps": "—",
                     "nb_correct_nat": stats["nb_correct"],
                     "nb_correct_adv": "—",
                     "nb_examples":    stats["nb_examples"],
                     "accuracy_nat":   stats["accuracy_nat"],
                     "accuracy_adv":   "—"})
    else:
        print("  WARNING: clean items could not be loaded, skipping.")

    # ---- adversarial evals -------------------------------------------------
    for threat_key in args.threat:
        cfg = THREAT_MODELS[threat_key]
        print(f"\n{'='*60}")
        print(f"  Split: {threat_key.upper()}  "
              f"norm={cfg['norm']}  eps={cfg['eps']}")

        if args.multi_gpu:
            stats = run_multi_gpu(clean_items, model_state, N,
                                  args.model_path, threat_key, args.batch_size)
        else:
            stats = run_single_gpu(clean_items, model, device,
                                   threat_key, args.batch_size)

        print(f"  Clean  accuracy (on this split's images): "
              f"{stats['nb_correct_nat']}/{stats['nb_examples']} "
              f"= {stats['accuracy_nat']:.4f}")
        print(f"  Robust accuracy: "
              f"{stats['nb_correct_adv']}/{stats['nb_examples']} "
              f"= {stats['accuracy_adv']:.4f}")
        rows.append({"split": threat_key, "norm": cfg["norm"], "eps": cfg["eps"],
                     "nb_correct_nat": stats["nb_correct_nat"],
                     "nb_correct_adv": stats["nb_correct_adv"],
                     "nb_examples":    stats["nb_examples"],
                     "accuracy_nat":   stats["accuracy_nat"],
                     "accuracy_adv":   stats["accuracy_adv"]})

    # ---- write CSV ---------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {csv_path}")

    # ---- summary table -----------------------------------------------------
    print(f"\n  {'Split':<8} {'Norm':<6} {'eps':<10} "
          f"{'Acc (nat)':>10} {'Acc (adv)':>10}")
    print(f"  {'-'*52}")
    for row in rows:
        print(f"  {row['split']:<8} {str(row['norm']):<6} {str(row['eps']):<10} "
              f"{str(row['accuracy_nat']):>10} {str(row['accuracy_adv']):>10}")
    print()

    if args.show_train_data:
        show_training_data_info(args.dataset)


if __name__ == "__main__":
    main()