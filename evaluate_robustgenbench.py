#!/usr/bin/env python3
"""
evaluate_robustgenbench.py
==========================
Evaluates a trained convnext model on all RobustGenBench adversarial test sets.

For each (surrogate × threat_model) combination it:
  1. Downloads the .tar.zst archive from HuggingFace (legolasflagstaff/RobustGenBench)
  2. Extracts images + labels.csv from the archive (in-memory, no persistent extraction)
  3. Loads the trained model state dict
  4. Runs distributed inference across all available GPUs (DDP-style via mp.spawn)
  5. Aggregates results and saves a CSV summary

Surrogates evaluated:
  - zeroshot_clip_vitb16_laion2b
  - zeroshot_clip_vith14_laion2b
  - zeroshot_metaclip_vith14_fullcc2_5b
  - zeroshot_siglip2_so400m_patch14_384

Threat models:
  - linf_eps4_autoattack_standard   (Linf 4/255)
  - l2_eps2_autoattack_standard     (L2 2.0)
  - l1_eps75_autoattack_standard    (L1 75.0)
  + common_severity3                (no surrogate)

Usage (called from job3_test_robustgenbench.sh):
  python evaluate_robustgenbench.py \
      --backbone convnext_base.fb_in22k \
      --dataset  flowers-102 \
      --loss     TRADES_v2 \
      --seed     1 \
      --project  convnext_base_fb_in22k_TRADES_v2
"""

import argparse
import csv
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Paths — adjust to match your cluster's layout (same vars as main codebase)
# ---------------------------------------------------------------------------

HF_REPO          = "legolasflagstaff/RobustGenBench"
HF_CACHE_DIR     = Path(os.path.expandvars(
    os.environ.get("HF_CACHE_DIR", "/tmp/robustgenbench/hf_cache")))
WORK_DIR         = Path(os.path.expandvars(
    os.environ.get("WORK_DIR", "/tmp/robustgenbench/work")))
RESULTS_BASE_DIR = Path(os.path.expandvars(
    os.environ.get("RESULTS_PATH", "$SCRATCH/results")))

# ---------------------------------------------------------------------------
# Evaluation matrix
# ---------------------------------------------------------------------------

SURROGATES = [
    "zeroshot_clip_vitb16_laion2b",
    "zeroshot_clip_vith14_laion2b",
    "zeroshot_metaclip_vith14_fullcc2_5b",
    "zeroshot_siglip2_so400m_patch14_384",
]

THREAT_MODELS = [
    "linf_eps4_autoattack_standard",
    "l2_eps2_autoattack_standard",
    "l1_eps75_autoattack_standard",
]

COMMON_SPLIT = "common/common_severity3"


def build_eval_matrix():
    """
    Returns a list of dicts, each describing one evaluation run:
      { hf_path, archive_name, surrogate, threat_model, label }
    """
    jobs = []

    # Surrogate × threat model combinations
    for surrogate in SURROGATES:
        for threat in THREAT_MODELS:
            jobs.append({
                "surrogate":    surrogate,
                "threat_model": threat,
                "hf_dir":       f"adversarial/{surrogate}/{threat}",
                "label":        f"{surrogate}__{threat}",
            })

    # Common corruptions (no surrogate)
    jobs.append({
        "surrogate":    "common",
        "threat_model": "common_severity3",
        "hf_dir":       "adversarial/common/common_severity3",
        "label":        "common__common_severity3",
    })

    return jobs


# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------

def find_archive_name(hf_dir: str, dataset: str) -> Optional[str]:
    """
    Finds the correct .tar.zst filename within hf_dir for a given dataset.
    Archive naming convention (from craft_adversarial.py):
      {dataset}__{surrogate_slug}__{threat_slug}_processed.tar.zst  (adversarial)
      {dataset}__common_severity3_processed.tar.zst                  (common)
    We list files in the HF directory and match by dataset prefix.
    """
    from huggingface_hub import list_repo_tree
    try:
        entries = list_repo_tree(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_in_repo=hf_dir,
        )
        for entry in entries:
            name = Path(entry.path).name
            if name.startswith(dataset) and name.endswith(".tar.zst"):
                return name
    except Exception as e:
        print(f"  WARNING: Could not list {hf_dir}: {e}")
    return None


def download_archive(hf_dir: str, archive_name: str, local_dir: Path) -> Optional[Path]:
    """
    Downloads {hf_dir}/{archive_name} from HF to local_dir.
    Returns local path, or None on failure.
    Skips download if file already exists.
    """
    from huggingface_hub import hf_hub_download

    local_path = local_dir / archive_name
    if local_path.exists():
        print(f"  Already cached: {archive_name}")
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)
    hf_path = f"{hf_dir}/{archive_name}"
    print(f"  Downloading: {hf_path}")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=hf_path,
            local_dir=str(local_dir),
            cache_dir=str(HF_CACHE_DIR),
        )
        # hf_hub_download may place the file in a subdirectory mirroring hf_path
        src = Path(downloaded)
        if src != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            src.rename(local_path)
        print(f"  Saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ERROR downloading {hf_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Archive extraction helpers
# ---------------------------------------------------------------------------

def load_archive_to_tmpdir(archive_path: Path, tmp_dir: Path) -> Optional[Path]:
    """
    Extracts a .tar.zst archive to tmp_dir.
    Returns the extracted directory path (tmp_dir/stem), or None on failure.
    """
    try:
        import zstandard as zstd
    except ImportError:
        print("ERROR: 'zstandard' package not installed. Run: pip install zstandard")
        sys.exit(1)

    stem     = archive_path.name.replace("_processed.tar.zst", "")
    dest_dir = tmp_dir / stem

    # Check if already extracted
    if (dest_dir / "test" / "labels.csv").exists():
        print(f"  Already extracted: {dest_dir.name}")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} → {dest_dir.name}/")
    try:
        with open(archive_path, "rb") as f_in:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f_in) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=dest_dir)
        return dest_dir
    except Exception as e:
        print(f"  ERROR extracting {archive_path.name}: {e}")
        return None


def read_labels_csv(extract_dir: Path) -> list:
    """
    Reads test/labels.csv from extracted archive.
    Returns list of (img_path: Path, label: int) tuples.
    """
    csv_path = extract_dir / "test" / "labels.csv"
    items    = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = extract_dir / "test" / row["filename"]
            label    = int(row["label"])
            items.append((img_path, label))
    return items


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RobustGenBenchDataset(Dataset):
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
    # Images in the archive are already 224×224 PNGs (saved by craft_adversarial.py)
    # We just convert to tensor; no normalization here — the model handles it internally.
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trained_model(args, N: int, rank: int):
    """
    Loads architecture + trained state dict onto the given rank/GPU.
    Reuses the same load_architecture + CustomModel pattern from the main codebase.
    """
    # Import from main codebase
    sys.path.insert(0, str(Path(__file__).parent))
    from architectures import load_architecture, CustomModel
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    # Build a minimal config for model construction
    # We load the optimal config from the HPO results
    hpo_source = args.hpo_source_project or args.project
    config_path = Path(args.configs_path) / "HPO_results" / hpo_source / f"{args.exp_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"HPO config not found at {config_path}. "
            f"Make sure training completed successfully."
        )
    config = OmegaConf.load(config_path)

    # Copy backbone .pt to work_path so load_architecture can find it
    import shutil
    backbone_src = Path(os.path.expanduser("~/links/scratch/mheuill/my_backbones")) / f"{config.backbone}.pt"
    backbone_dst = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    backbone_dst.mkdir(parents=True, exist_ok=True)
    if backbone_src.exists():
        shutil.copy2(str(backbone_src), str(backbone_dst))
        print(f"  Copied backbone weights to {backbone_dst}")
    else:
        print(f"  WARNING: backbone not found at {backbone_src}")

    model = load_architecture(config, N)
    model = CustomModel(config, model)

    # Load trained weights
    state_dict_path = Path(args.trained_statedicts_path) / args.project / f"{args.exp_id}.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict not found at {state_dict_path}")

    print(f"  Loading weights from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(rank)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Distributed inference worker
# ---------------------------------------------------------------------------

def inference_worker(rank, world_size, items, args, result_queue, N):
    """
    Each process evaluates its shard of the dataset.
    Puts (rank, nb_correct, nb_examples) into result_queue.
    """
    # Init process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = load_trained_model(args, N, rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    transform = build_transform()
    dataset   = RobustGenBenchDataset(items, transform)
    sampler   = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                   shuffle=False, drop_last=False)
    loader    = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                           num_workers=4, pin_memory=True)

    nb_correct  = 0
    nb_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(rank)
            labels = labels.to(rank)

            # CustomModel forward: pass same tensor twice (no adversarial generation here)
            logits_nat, _ = model(images, images)
            preds = logits_nat.argmax(dim=1)

            nb_correct  += (preds == labels).sum().item()
            nb_examples += labels.size(0)

    result_queue.put((rank, nb_correct, nb_examples))
    dist.barrier()
    dist.destroy_process_group()


def run_distributed_inference(items, args, N) -> dict:
    """
    Spawns world_size processes, aggregates accuracy.
    Returns { 'nb_correct': int, 'nb_examples': int, 'accuracy': float }
    """
    world_size   = torch.cuda.device_count()
    result_queue = mp.Queue()

    mp.spawn(
        inference_worker,
        args=(world_size, items, args, result_queue, N),
        nprocs=world_size,
        join=True,
    )

    total_correct  = 0
    total_examples = 0
    while not result_queue.empty():
        rank, nb_correct, nb_examples = result_queue.get()
        total_correct  += nb_correct
        total_examples += nb_examples

    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    return {
        "nb_correct":  total_correct,
        "nb_examples": total_examples,
        "accuracy":    round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--loss",       required=True)
    parser.add_argument("--seed",       type=int, default=1)
    parser.add_argument("--project",    required=True,
                        help="Project name where trained model is saved")
    parser.add_argument("--hpo_source_project", default=None,
                        help="Project name to load HPO yaml from. Defaults to --project if not set.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--configs_path",
                        default=os.environ.get("CONFIGS_PATH", "./configs"))
    parser.add_argument("--trained_statedicts_path",
                        default=os.environ.get("TRAINED_STATEDICTS_PATH",
                                               "$SCRATCH/trained_statedicts"))
    parser.add_argument("--results_path",
                        default=os.environ.get("RESULTS_PATH",
                                               "$SCRATCH/results"))
    args = parser.parse_args()

    # Resolve env vars in paths
    args.trained_statedicts_path = os.path.expandvars(
        os.path.expanduser(args.trained_statedicts_path))
    args.results_path = os.path.expandvars(
        os.path.expanduser(args.results_path))
    args.configs_path = os.path.expandvars(
        os.path.expanduser(args.configs_path))

    # exp_id follows the same convention as distributed_experiment_final.py
    args.exp_id = f"{args.backbone}__{args.dataset}__{args.loss}"

    print("=" * 60)
    print(f"  RobustGenBench Evaluation")
    print(f"  Backbone  : {args.backbone}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Loss      : {args.loss}")
    print(f"  Project   : {args.project}")
    print(f"  exp_id    : {args.exp_id}")
    print(f"  GPUs      : {torch.cuda.device_count()}")
    print("=" * 60)

    # Number of classes — load from the HPO config
    from omegaconf import OmegaConf
    hpo_source = args.hpo_source_project or args.project
    config_path = Path(args.configs_path) / "HPO_results" / hpo_source / f"{args.exp_id}.yaml"
    config      = OmegaConf.load(config_path)
    # Download class_names json from HF to get num_classes
    import json
    from huggingface_hub import hf_hub_download
    class_names_file = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"class_names/{args.dataset}.json",
        local_dir=str(WORK_DIR / "class_names"),
        cache_dir=str(HF_CACHE_DIR),
    )
    with open(class_names_file) as f:
        N = len(json.load(f))
    print(f"  num_classes={N} (from class_names/{args.dataset}.json)")

    # Output CSV
    results_dir = Path(args.results_path) / args.project
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{args.exp_id}__robustgenbench_results.csv"

    # Load existing results to allow resuming
    completed_labels = set()
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_labels.add(row["label"])
        print(f"  Resuming: {len(completed_labels)} evaluations already done.")

    # Local cache dirs
    archives_dir  = WORK_DIR / "archives"  / args.dataset
    extracted_dir = WORK_DIR / "extracted" / args.dataset
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    eval_matrix = build_eval_matrix()

    # Write CSV header if new file
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    fieldnames = ["dataset", "project", "backbone", "loss", "surrogate",
                  "threat_model", "label", "nb_correct", "nb_examples", "accuracy"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    torch.multiprocessing.set_start_method("spawn", force=True)

    for job in eval_matrix:
        label = job["label"]

        if label in completed_labels:
            print(f"\n  [SKIP] {label} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {label}")
        print(f"  HF dir    : {job['hf_dir']}")

        # 1. Find archive name
        archive_name = find_archive_name(job["hf_dir"], args.dataset)
        if archive_name is None:
            print(f"  WARNING: No archive found for dataset={args.dataset} "
                  f"in {job['hf_dir']}. Skipping.")
            continue

        # 2. Download archive
        archive_path = download_archive(job["hf_dir"], archive_name, archives_dir)
        if archive_path is None:
            print(f"  WARNING: Download failed for {archive_name}. Skipping.")
            continue

        # 3. Extract
        extract_dir = load_archive_to_tmpdir(archive_path, extracted_dir)
        if extract_dir is None:
            print(f"  WARNING: Extraction failed for {archive_name}. Skipping.")
            continue

        # 4. Load items
        items = read_labels_csv(extract_dir)
        print(f"  Loaded {len(items)} test images.")

        # 5. Run distributed inference
        print(f"  Running inference on {torch.cuda.device_count()} GPUs...")
        stats = run_distributed_inference(items, args, N)

        print(f"  Result: {stats['nb_correct']}/{stats['nb_examples']} "
              f"= {stats['accuracy']:.4f}")

        # 6. Write result row
        writer.writerow({
            "dataset":      args.dataset,
            "project":      args.project,
            "backbone":     args.backbone,
            "loss":         args.loss,
            "surrogate":    job["surrogate"],
            "threat_model": job["threat_model"],
            "label":        label,
            "nb_correct":   stats["nb_correct"],
            "nb_examples":  stats["nb_examples"],
            "accuracy":     stats["accuracy"],
        })
        csv_file.flush()
        completed_labels.add(label)

    csv_file.close()
    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()