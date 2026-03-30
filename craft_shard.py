#!/usr/bin/env python3
"""
craft_shard.py — Run craft_adversarial.py on a subset of a dataset (one GPU),
then optionally merge all shards into the main output and package/upload.

Shard mode (called once per GPU):
    python craft_shard.py --dataset caltech101 --norm L1 --eps 75
        --surrogate clip_vith14 --batch_size 128
        --shard_idx 0 --n_shards 4 --gpu 0

Merge mode (called once after all shards finish):
    python craft_shard.py --dataset caltech101 --norm L1 --eps 75
        --surrogate clip_vith14 --merge --upload_hf
"""

import argparse
import csv
import io
import json
import os
import sys
import tarfile
from pathlib import Path

# Reuse helpers from craft_adversarial
sys.path.insert(0, str(Path(__file__).parent))
from craft_adversarial import (
    OUTPUT_ROOT, PACKAGED_ROOT, DATA_ROOT, WORK_DIR, HF_CACHE_DIR,
    TMP_ROOT, CLASS_NAMES_DIR,
    run_dir_name, surrogate_slug, threat_model_slug, hf_archive_path,
    eps_to_float, load_surrogate, build_transform, AdversarialDataset,
    extract_archive, load_local_dataset, load_class_names,
    save_batch, package_run, upload_to_hf, ensure_data_downloaded,
)

import numpy as np
import torch
from autoattack import AutoAttack
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_shard(args):
    dataset   = args.dataset
    norm      = args.norm
    eps       = float(args.eps)
    eps_float = eps_to_float(norm, eps)
    shard_idx = args.shard_idx
    n_shards  = args.n_shards

    # Each shard writes to its own subdirectory to avoid conflicts
    base_rname = run_dir_name(dataset, args.surrogate, norm, eps)
    shard_dir  = OUTPUT_ROOT / f"{base_rname}__shard{shard_idx}"
    done_flag  = shard_dir / "shard_done.json"

    if done_flag.exists():
        print(f"  Shard {shard_idx}/{n_shards} already complete — skipping")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset   : {dataset}  [shard {shard_idx+1}/{n_shards}]")
    print(f"  Surrogate : {args.surrogate}")
    print(f"  Norm      : {norm}  eps={eps}  (AA: {eps_float:.5f})")
    print(f"  GPU       : {args.gpu}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    dataset_dir   = extract_archive(dataset)
    all_items     = load_local_dataset(dataset_dir, split="test")
    label_to_name = load_class_names(dataset)

    # Split indices across shards
    total      = len(all_items)
    shard_size = (total + n_shards - 1) // n_shards
    start      = shard_idx * shard_size
    end        = min(start + shard_size, total)
    items      = all_items[start:end]

    print(f"  Shard indices: {start}–{end-1}  ({len(items)} samples of {total} total)")

    model     = load_surrogate(args.surrogate, label_to_name, device, dataset=dataset)
    transform = build_transform()
    ds        = AdversarialDataset(items, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    shard_dir.mkdir(parents=True, exist_ok=True)

    adversary = AutoAttack(
        model, norm=norm, eps=eps_float,
        version="standard", device=device, verbose=True,
    )

    n_correct_clean = n_correct_adv = n_total = 0
    meta_file = open(shard_dir / "metadata.jsonl", "a")

    for x, labels, filenames in tqdm(loader, desc=f"shard{shard_idx}"):
        x        = x.to(device)
        labels_t = torch.tensor(labels, dtype=torch.long).to(device) \
                   if not isinstance(labels, torch.Tensor) else labels.to(device)

        with torch.no_grad():
            n_correct_clean += (model(x).argmax(1) == labels_t).sum().item()

        x_adv = adversary.run_standard_evaluation(x, labels_t, bs=x.size(0))

        with torch.no_grad():
            n_correct_adv += (model(x_adv).argmax(1) == labels_t).sum().item()

        n_total += x.size(0)
        records = save_batch(x_adv, labels_t.cpu().tolist(), filenames,
                             label_to_name, shard_dir)
        for rec in records:
            meta_file.write(json.dumps(rec) + "\n")
        meta_file.flush()

    meta_file.close()

    done_flag.write_text(json.dumps({
        "shard_idx": shard_idx,
        "n_shards":  n_shards,
        "start":     start,
        "end":       end,
        "n_total":   n_total,
        "clean_acc": round(n_correct_clean / n_total, 4),
        "adv_acc":   round(n_correct_adv   / n_total, 4),
    }, indent=2))

    print(f"\n  Shard {shard_idx} done: {n_total} images | "
          f"clean={n_correct_clean/n_total:.4f} | adv={n_correct_adv/n_total:.4f}")


def merge_shards(args):
    dataset  = args.dataset
    norm     = args.norm
    eps      = float(args.eps)

    base_rname = run_dir_name(dataset, args.surrogate, norm, eps)
    output_dir = OUTPUT_ROOT / base_rname
    done_flag  = output_dir / "surrogate_summary.json"

    if done_flag.exists():
        print(f"  Already merged — skipping merge")
        if args.upload_hf:
            archive_path = package_run(output_dir, PACKAGED_ROOT)
            upload_to_hf(archive_path, args.surrogate, norm, eps)
        return

    # Collect all shard dirs
    shard_dirs = sorted(OUTPUT_ROOT.glob(f"{base_rname}__shard*"))
    if not shard_dirs:
        print(f"  ERROR: no shard directories found for {base_rname}")
        sys.exit(1)

    print(f"  Merging {len(shard_dirs)} shards → {output_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    n_clean = n_adv = n_total = 0

    for sd in shard_dirs:
        done_path = sd / "shard_done.json"
        if not done_path.exists():
            print(f"  WARNING: shard {sd.name} not complete (no shard_done.json) — skipping")
            continue
        info = json.loads(done_path.read_text())
        n_total += info["n_total"]
        n_clean += round(info["clean_acc"] * info["n_total"])
        n_adv   += round(info["adv_acc"]   * info["n_total"])

        meta_path = sd / "metadata.jsonl"
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    all_records.append(rec)

        # Copy images
        for img_file in sd.glob("*.png"):
            dest = output_dir / img_file.name
            if not dest.exists():
                import shutil
                shutil.copy2(img_file, dest)

    # Write merged metadata.jsonl
    with open(output_dir / "metadata.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    clean_acc = n_clean / n_total if n_total else 0
    adv_acc   = n_adv   / n_total if n_total else 0

    done_flag.write_text(json.dumps({
        "n_total":             n_total,
        "surrogate_clean_acc": round(clean_acc, 4),
        "surrogate_adv_acc":   round(adv_acc,   4),
        "attack_success_rate": round(1 - adv_acc, 4),
        "n_shards_merged":     len(shard_dirs),
    }, indent=2))

    print(f"  Merged {n_total} images | clean={clean_acc:.4f} | adv={adv_acc:.4f}")

    archive_path = package_run(output_dir, PACKAGED_ROOT)
    if args.upload_hf:
        upload_to_hf(archive_path, args.surrogate, norm, eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--norm",       required=True)
    p.add_argument("--eps",        type=float, required=True)
    p.add_argument("--surrogate",  default="clip_vith14")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_shards",   type=int, default=4)
    p.add_argument("--shard_idx",  type=int, default=None)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--merge",      action="store_true")
    p.add_argument("--upload_hf",  action="store_true")

    args = p.parse_args()

    for d in [TMP_ROOT, DATA_ROOT, HF_CACHE_DIR, OUTPUT_ROOT, PACKAGED_ROOT, WORK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    ensure_data_downloaded()

    if args.merge:
        merge_shards(args)
    elif args.shard_idx is not None:
        run_shard(args)
    else:
        print("ERROR: specify --shard_idx N or --merge")
        sys.exit(1)


if __name__ == "__main__":
    main()