#!/usr/bin/env python3
"""
evaluate_robustgenbench.py
==========================
Evaluates a trained convnext model on all RobustGenBench adversarial test sets.

Usage (called from job3_test_robustgenbench.sh):
  python evaluate_robustgenbench.py \
      --backbone convnext_base.fb_in22k \
      --dataset  flowers-102 \
      --loss     TRADES_v2 \
      --seed     1 \
      --project  convnext_base_fb_in22k_TRADES_v2 \
      --hpo_source_project full_fine_tuning50
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HF_REPO      = "legolasflagstaff/RobustGenBench"
HF_CACHE_DIR = Path(os.path.expandvars(os.environ.get("HF_HOME", "/tmp/hf_cache")))
WORK_DIR     = Path(os.path.expandvars(os.environ.get("WORK_DIR", "/tmp/work")))

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


def build_eval_matrix():
    jobs = []
    for surrogate in SURROGATES:
        for threat in THREAT_MODELS:
            jobs.append({
                "surrogate":    surrogate,
                "threat_model": threat,
                "hf_dir":       f"adversarial/{surrogate}/{threat}",
                "label":        f"{surrogate}__{threat}",
            })
    jobs.append({
        "surrogate":    "common",
        "threat_model": "common_severity3",
        "hf_dir":       "adversarial/common/common_severity3",
        "label":        "common__common_severity3",
    })
    return jobs


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def find_archive_name(hf_dir: str, dataset: str) -> Optional[str]:
    from huggingface_hub import list_repo_tree
    try:
        entries = list_repo_tree(repo_id=HF_REPO, repo_type="dataset", path_in_repo=hf_dir)
        for entry in entries:
            name = Path(entry.path).name
            if name.startswith(dataset) and name.endswith(".tar.zst"):
                return name
    except Exception as e:
        print(f"  WARNING: Could not list {hf_dir}: {e}")
    return None


def download_archive(hf_dir: str, archive_name: str, local_dir: Path) -> Optional[Path]:
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
            repo_id=HF_REPO, repo_type="dataset", filename=hf_path,
            local_dir=str(local_dir), cache_dir=str(HF_CACHE_DIR),
        )
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
# Archive extraction
# ---------------------------------------------------------------------------

def load_archive_to_tmpdir(archive_path: Path, tmp_dir: Path) -> Optional[Path]:
    try:
        import zstandard as zstd
    except ImportError:
        print("ERROR: zstandard not installed"); sys.exit(1)

    stem     = archive_path.name.replace("_processed.tar.zst", "")
    dest_dir = tmp_dir / stem

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
    csv_path = extract_dir / "test" / "labels.csv"
    items = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = extract_dir / "test" / row["filename"]
            items.append((img_path, int(row["label"])))
    return items


# ---------------------------------------------------------------------------
# Clean test set loader
# ---------------------------------------------------------------------------

def load_clean_test_items(args) -> Optional[list]:
    from huggingface_hub import hf_hub_download
    archive_name = f"{args.dataset}_processed.tar.zst"
    clean_archives_dir = WORK_DIR / "clean_archives"
    clean_extract_dir  = WORK_DIR / "clean"
    sentinel = clean_extract_dir / archive_name.replace("_processed.tar.zst", "") / "test" / "labels.csv"

    if not sentinel.exists():
        try:
            print(f"  Downloading clean test archive: {archive_name}")
            downloaded = hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset",
                filename=archive_name,
                local_dir=str(clean_archives_dir),
                cache_dir=str(HF_CACHE_DIR),
            )
            extract_dir = load_archive_to_tmpdir(Path(downloaded), clean_extract_dir)
            if extract_dir is None:
                return None
        except Exception as e:
            print(f"  WARNING: Could not load clean test set: {e}")
            return None

    dataset_dir = clean_extract_dir / args.dataset.replace("_processed", "")
    # find the extracted folder
    candidates = list(clean_extract_dir.glob(f"{args.dataset}*"))
    if not candidates:
        print(f"  WARNING: Could not find extracted clean dir for {args.dataset}")
        return None
    return read_labels_csv(candidates[0])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RobustGenBenchDataset(Dataset):
    def __init__(self, items, transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label, idx


def build_transform():
    # Images are already 224×224; just convert to tensor.
    # CustomModel handles normalization internally.
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trained_model(args, N: int, rank: int):
    import shutil
    sys.path.insert(0, str(Path(__file__).parent))
    from architectures import load_architecture, CustomModel
    from omegaconf import OmegaConf

    # Load HPO config from hpo_source_project (may differ from project)
    hpo_source  = args.hpo_source_project or args.project
    config_path = Path(args.configs_path) / "HPO_results" / hpo_source / f"{args.exp_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"HPO config not found at {config_path}")
    config = OmegaConf.load(config_path)

    # Copy backbone .pt to work_path/SLURM_JOB_ID so load_architecture can find
    # it (must match the job-scoped statedict_dir it reads from — see
    # architectures/loaders.py — to avoid concurrent-job path collisions).
    backbone_src = Path(os.path.expanduser("~/links/scratch/mheuill/my_backbones")) / f"{config.backbone}.pt"
    backbone_dst = Path(os.path.expandvars(config.work_path)).expanduser().resolve() / os.environ.get("SLURM_JOB_ID", "local")
    backbone_dst.mkdir(parents=True, exist_ok=True)
    if backbone_src.exists():
        shutil.copy2(str(backbone_src), str(backbone_dst))
    else:
        print(f"  WARNING: backbone not found at {backbone_src}")

    model = load_architecture(config, N)
    model = CustomModel(config, model)

    # Load trained weights. distributed_experiment_final.py's training() saves
    # to f"{config.exp_id}.pt" where config is config_optimal (loaded fresh from
    # the HPO yaml) — only config_optimal.project_name gets overridden there,
    # never .exp_id, so the filename actually used is the yaml's OWN stored
    # exp_id field (single "_" separator, stale from HPO-save time), NOT
    # args.exp_id (double "__", only used to locate the yaml file itself).
    state_dict_filename = f"{args.backbone}_{args.dataset}_{args.loss}.pt"
    state_dict_path = Path(os.path.expanduser(args.trained_statedicts_path)) / args.project / state_dict_filename
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict not found at {state_dict_path}")

    print(f"  Loading weights from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(rank)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Distributed inference
# ---------------------------------------------------------------------------

def inference_worker(rank, world_size, items, args, result_queue, N):
    os.environ["MASTER_ADDR"] = "localhost"
    # Derived from SLURM_JOB_ID rather than a fixed default: concurrent eval
    # jobs on the same node would otherwise all rendezvous on the same port.
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(20000 + (int(job_id) % 20000)))
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
    records     = []  # (idx, true_label, pred_label), only populated if args.save_predictions

    with torch.no_grad():
        for images, labels, idxs in loader:
            images = images.to(rank)
            labels = labels.to(rank)
            logits_nat, _ = model(images, images)
            preds = logits_nat.argmax(dim=1)
            nb_correct  += (preds == labels).sum().item()
            nb_examples += labels.size(0)
            if args.save_predictions:
                records.extend(zip(idxs.tolist(), labels.tolist(), preds.tolist()))

    result_queue.put((rank, nb_correct, nb_examples, records))
    dist.barrier()
    dist.destroy_process_group()


def run_distributed_inference(items, args, N):
    world_size   = torch.cuda.device_count()
    result_queue = mp.Queue()
    mp.spawn(inference_worker, args=(world_size, items, args, result_queue, N),
             nprocs=world_size, join=True)
    total_correct  = 0
    total_examples = 0
    # DistributedSampler(drop_last=False) pads the last shard by repeating a few
    # leading indices across ranks; dedupe by idx so per-observation predictions
    # contain exactly one row per dataset item (required for valid bootstrapping).
    by_idx = {}
    while not result_queue.empty():
        rank, nb_correct, nb_examples, records = result_queue.get()
        total_correct  += nb_correct
        total_examples += nb_examples
        for idx, true_label, pred_label in records:
            by_idx[idx] = (true_label, pred_label)
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    stats = {"nb_correct": total_correct, "nb_examples": total_examples,
             "accuracy": round(accuracy, 4)}
    predictions = (sorted((idx, tl, pl) for idx, (tl, pl) in by_idx.items())
                   if args.save_predictions else None)
    return stats, predictions


def write_predictions_csv(records, items, results_dir: Path, exp_id: str, label: str):
    preds_dir = results_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out_path = preds_dir / f"{exp_id}__{label}__predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "filename", "true_label", "pred_label", "correct"])
        for idx, true_label, pred_label in records:
            filename = items[idx][0].name
            writer.writerow([idx, filename, true_label, pred_label,
                              int(true_label == pred_label)])
    print(f"  Predictions saved to: {out_path} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",             required=True)
    parser.add_argument("--dataset",              required=True)
    parser.add_argument("--loss",                 required=True)
    parser.add_argument("--seed",                 type=int, default=1)
    parser.add_argument("--project",              required=True)
    parser.add_argument("--hpo_source_project",   default=None)
    parser.add_argument("--batch_size",           type=int, default=256)
    parser.add_argument("--save_predictions",     action="store_true",
                        help="Also save per-observation predictions CSV for bootstrap analysis.")
    parser.add_argument("--configs_path",
                        default=os.environ.get("CONFIGS_PATH", "./configs"))
    parser.add_argument("--trained_statedicts_path",
                        default=os.environ.get("TRAINED_STATEDICTS_PATH",
                                               "~/scratch/trained_state_dicts"))
    parser.add_argument("--results_path",
                        default=os.environ.get("ROBUSTGENBENCH_RESULTS_PATH",
                                               "./robustgenbench_eval"))
    args = parser.parse_args()

    # Resolve paths
    args.trained_statedicts_path = os.path.expandvars(os.path.expanduser(args.trained_statedicts_path))
    args.results_path            = os.path.expandvars(os.path.expanduser(args.results_path))
    args.configs_path            = os.path.expandvars(os.path.expanduser(args.configs_path))

    # exp_id uses __ separator to match yaml filenames
    args.exp_id = f"{args.backbone}__{args.dataset}__{args.loss}"

    print("=" * 60)
    print(f"  RobustGenBench Evaluation")
    print(f"  Backbone         : {args.backbone}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Loss             : {args.loss}")
    print(f"  Project          : {args.project}")
    print(f"  HPO source       : {args.hpo_source_project or args.project}")
    print(f"  exp_id           : {args.exp_id}")
    print(f"  GPUs             : {torch.cuda.device_count()}")
    print("=" * 60)

    # Get num_classes from HF class_names json
    from huggingface_hub import hf_hub_download
    import json
    class_names_file = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"class_names/{args.dataset}.json",
        local_dir=str(WORK_DIR / "class_names"),
        cache_dir=str(HF_CACHE_DIR),
    )
    with open(class_names_file) as f:
        N = len(json.load(f))
    print(f"  num_classes={N}")

    # Output CSV — always overwrite for fresh results
    results_dir = Path(args.results_path) / args.project
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{args.exp_id}__robustgenbench_results.csv"
    if csv_path.exists():
        csv_path.unlink()
        print(f"  Deleted existing CSV for fresh evaluation.")

    # Local cache dirs
    adv_archives_base = Path(os.path.expandvars(
        os.environ.get("ADV_ARCHIVES_PATH", str(WORK_DIR / "adv_archives"))))
    archives_dir  = adv_archives_base / args.dataset
    extracted_dir = WORK_DIR / "extracted" / args.dataset
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "project", "backbone", "loss", "surrogate",
                  "threat_model", "label", "nb_correct", "nb_examples", "accuracy"]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    torch.multiprocessing.set_start_method("spawn", force=True)

    # --- Clean accuracy ---
    print(f"\n{'='*60}")
    print(f"  Evaluating: CLEAN")
    clean_items = load_clean_test_items(args)
    if clean_items is not None:
        print(f"  Loaded {len(clean_items)} clean images.")
        # Pixel sanity check
        arr = np.array(Image.open(clean_items[0][0]).convert("RGB")) / 255.0
        print(f"  Clean img[0]: min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
        stats, preds = run_distributed_inference(clean_items, args, N)
        print(f"  Clean accuracy: {stats['nb_correct']}/{stats['nb_examples']} = {stats['accuracy']:.4f}")
        writer.writerow({"dataset": args.dataset, "project": args.project,
                         "backbone": args.backbone, "loss": args.loss,
                         "surrogate": "none", "threat_model": "clean",
                         "label": "clean", "nb_correct": stats["nb_correct"],
                         "nb_examples": stats["nb_examples"], "accuracy": stats["accuracy"]})
        csv_file.flush()
        if args.save_predictions and preds:
            write_predictions_csv(preds, clean_items, results_dir, args.exp_id, "clean")

    # --- Adversarial evaluations ---
    for job in build_eval_matrix():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {job['label']}")

        archive_name = find_archive_name(job["hf_dir"], args.dataset)
        if archive_name is None:
            print(f"  WARNING: No archive found. Skipping.")
            continue

        archive_path = download_archive(job["hf_dir"], archive_name, archives_dir)
        if archive_path is None:
            print(f"  WARNING: Download failed. Skipping.")
            continue

        extract_dir = load_archive_to_tmpdir(archive_path, extracted_dir)
        if extract_dir is None:
            print(f"  WARNING: Extraction failed. Skipping.")
            continue

        items = read_labels_csv(extract_dir)
        print(f"  Loaded {len(items)} images.")

        # Pixel sanity check
        arr = np.array(Image.open(items[0][0]).convert("RGB")) / 255.0
        print(f"  Adv img[0]: min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")

        stats, preds = run_distributed_inference(items, args, N)
        print(f"  Result: {stats['nb_correct']}/{stats['nb_examples']} = {stats['accuracy']:.4f}")

        writer.writerow({"dataset": args.dataset, "project": args.project,
                         "backbone": args.backbone, "loss": args.loss,
                         "surrogate": job["surrogate"], "threat_model": job["threat_model"],
                         "label": job["label"], "nb_correct": stats["nb_correct"],
                         "nb_examples": stats["nb_examples"], "accuracy": stats["accuracy"]})
        csv_file.flush()
        if args.save_predictions and preds:
            write_predictions_csv(preds, items, results_dir, args.exp_id, job["label"])

    csv_file.close()
    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()