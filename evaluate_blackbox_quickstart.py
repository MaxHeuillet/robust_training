#!/usr/bin/env python3
"""
evaluate_blackbox_quickstart.py
================================
Black-box adversarial evaluation on RobustGenBench.

This script summarizes the full black-box evaluation methodology:

  1. SURROGATE  — a small CLIP model (openai/clip-vit-base-patch16) is used
                  to craft adversarial examples with AutoAttack (L-inf, white-box
                  on the surrogate).
  2. TRANSFER   — the adversarial examples are sent to an API-based LLM target
                  (OpenAI GPT-4o-mini) via the Batch API to measure transfer
                  robustness. Clean images are also submitted to get the baseline.

This reflects the benchmark's black-box threat model: the attacker has
white-box access to a surrogate but only black-box (API) access to the target.

Dependencies:
    pip install torch torchvision huggingface_hub zstandard Pillow numpy
    pip install git+https://github.com/fra31/auto-attack
    pip install transformers openai

    Set your OpenAI key:
        export OPENAI_API_KEY=sk-...

Usage:
    # Default: flowers-102, L-inf eps=30/255, 64 images, gpt-4o-mini
    python evaluate_blackbox_quickstart.py --submit

    # Retrieve results after the batch completes (batch_id printed on submit)
    python evaluate_blackbox_quickstart.py --retrieve <batch_id>

    # Full loop over all benchmark datasets
    python evaluate_blackbox_quickstart.py --submit --all-datasets

    # Override defaults
    python evaluate_blackbox_quickstart.py --submit \\
        --dataset flowers-102 \\
        --eps 0.05 \\
        --n-images 128 \\
        --model gpt-4o
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO      = "legolasflagstaff/RobustGenBench"
HF_CACHE_DIR = Path(os.environ.get("HF_HOME",   "/tmp/hf_cache"))
WORK_DIR     = Path(os.environ.get("WORK_DIR",   "/tmp/robustgenbench_blackbox"))

# Default attack parameters — L-inf at 30/255 as used in the paper
DEFAULT_NORM    = "Linf"
DEFAULT_EPS     = 30 / 255
DEFAULT_DATASET = "flowers-102"
DEFAULT_N       = 64          # images per dataset in quickstart mode
DEFAULT_MODEL   = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# CLIP surrogate
# ---------------------------------------------------------------------------

class CLIPSurrogate(nn.Module):
    """
    Wraps openai/clip-vit-base-patch16 as a plain image classifier:
        model(images: Tensor[B,3,224,224]) -> logits: Tensor[B, N]

    Zero-shot classification is done by computing cosine similarity between
    image embeddings and the text embeddings of each class name.
    Text embeddings are computed once and cached.
    Inputs are expected in [0, 1]; CLIP preprocessing is applied internally.
    """
    def __init__(self, class_names: list[str], device: torch.device):
        super().__init__()
        from transformers import CLIPProcessor, CLIPModel
        print("  Loading CLIP surrogate (openai/clip-vit-base-patch16) ...")
        self.clip      = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.device    = device
        self.clip.to(device)
        self.clip.eval()

        # Cache text embeddings for all class names
        prompts = [f"a photo of a {name}" for name in class_names]
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt",
                                    padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_emb = self.clip.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        # Register as buffer so it moves with the model
        self.register_buffer("text_embeddings", text_emb)

        # CLIP pixel normalization (applied to [0,1] inputs)
        self.register_buffer("mean", torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize from [0,1] to CLIP's expected range
        x = (x - self.mean) / self.std
        image_emb = self.clip.get_image_features(pixel_values=x)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        # Cosine similarity as logits, scaled by CLIP's temperature
        logits = 100.0 * image_emb @ self.text_embeddings.T
        return logits

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
        return self.transform(img), label, str(img_path)


def build_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),   # -> [0, 1]
    ])

# ---------------------------------------------------------------------------
# Attack: craft adversarial examples on the CLIP surrogate
# ---------------------------------------------------------------------------

def craft_adversarial_examples(
    surrogate: CLIPSurrogate,
    items: list,
    device: torch.device,
    norm: str,
    eps: float,
    batch_size: int,
) -> list[tuple[np.ndarray, int, str]]:
    """
    Run AutoAttack on the CLIP surrogate (white-box) and return a list of
    (adv_image_uint8, label, original_path) tuples ready for API submission.

    This mirrors the attack loop in distributed_experiment_final.py:
      adversary = AutoAttack(forward_pass, norm=norm, eps=eps, ...)
      x_adv = adversary.run_standard_evaluation(x_nat, target, bs=batch_size)
    """
    from autoattack import AutoAttack

    transform = build_transform()
    loader    = DataLoader(BenchmarkDataset(items, transform),
                           batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    adversary = AutoAttack(surrogate, norm=norm, eps=eps,
                           version="standard", verbose=True, device=device)

    results = []
    surrogate.eval()
    for x_nat, labels, paths in loader:
        x_nat   = x_nat.to(device)
        labels  = labels.to(device)
        x_adv   = adversary.run_standard_evaluation(x_nat, labels, bs=batch_size)

        # Convert adversarial tensors to uint8 numpy for API submission
        x_adv_np = (x_adv.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        for i in range(x_adv_np.shape[0]):
            # Channel-first -> HWC
            img_hwc = x_adv_np[i].transpose(1, 2, 0)
            results.append((img_hwc, labels[i].item(), paths[i]))

    return results

# ---------------------------------------------------------------------------
# HuggingFace helpers  (mirrors evaluate_robustgenbench.py)
# ---------------------------------------------------------------------------

def get_class_names(dataset: str) -> dict[int, str]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"class_names/{dataset}.json",
        local_dir=str(WORK_DIR / "class_names"),
        cache_dir=str(HF_CACHE_DIR),
    )
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    return {i: v for i, v in enumerate(raw)}


def list_datasets() -> list[str]:
    from huggingface_hub import list_repo_tree
    datasets = []
    try:
        for entry in list_repo_tree(repo_id=HF_REPO, repo_type="dataset",
                                    path_in_repo="class_names"):
            name = Path(entry.path).name
            if name.endswith(".json"):
                datasets.append(name[:-5])
    except Exception as e:
        print(f"  WARNING: could not list datasets: {e}")
    return sorted(datasets)


def download_archive(filename: str, local_dir: Path) -> Path | None:
    from huggingface_hub import hf_hub_download
    local_path = local_dir / filename
    if local_path.exists():
        return local_path
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {filename}")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            cache_dir=str(HF_CACHE_DIR),
        )
        src = Path(downloaded)
        if src != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            src.rename(local_path)
        return local_path
    except Exception as e:
        print(f"  ERROR: {e}")
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


def load_test_items(dataset: str, n: int | None = None) -> list | None:
    archive_name = f"{dataset}_processed.tar.zst"
    archives_dir = WORK_DIR / "clean_archives"
    extract_root = WORK_DIR / "clean"

    archive_path = download_archive(archive_name, archives_dir)
    if archive_path is None:
        return None

    extract_dir = extract_archive(archive_path, extract_root)
    if extract_dir is None:
        return None

    csv_path = extract_dir / "test" / "labels.csv"
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append((extract_dir / "test" / row["filename"], int(row["label"])))

    if n is not None:
        items = items[:n]
    return items

# ---------------------------------------------------------------------------
# OpenAI Batch API helpers
# ---------------------------------------------------------------------------

def image_to_base64(img_hwc: np.ndarray) -> str:
    """Encode a HWC uint8 numpy array as a base64 JPEG string."""
    pil = Image.fromarray(img_hwc)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_classification_prompt(class_names: list[str]) -> str:
    """System prompt that constrains the LLM to output only a class name."""
    names_str = "\n".join(f"- {n}" for n in class_names)
    return (
        "You are an image classifier. Given an image, output ONLY the name "
        "of the most likely class from the following list — nothing else, no "
        "punctuation, no explanation:\n\n" + names_str
    )


def submit_batch(
    adv_results:  list[tuple[np.ndarray, int, str]],
    clean_items:  list[tuple[Path, int]],
    class_names:  list[str],
    label_to_name: dict[int, str],
    openai_model: str,
    run_id:       str,
    output_dir:   Path,
) -> str:
    """
    Build an OpenAI batch JSONL (clean + adversarial images interleaved),
    upload it, and submit the batch. Returns the batch_id.

    Each request carries a custom_id encoding:
        <run_id>__clean__<idx>__label_<label>
        <run_id>__adv__<idx>__label_<label>
    so results can be matched back to ground-truth labels on retrieval.
    """
    import openai
    client = openai.OpenAI()

    system_prompt = build_classification_prompt(class_names)
    requests      = []

    def make_request(custom_id: str, b64: str) -> dict:
        return {
            "custom_id": custom_id,
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model": openai_model,
                "max_tokens": 32,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                       "detail": "low"}},
                        {"type": "text", "text": "What class is this image?"},
                    ]},
                ],
            },
        }

    # Clean images
    for idx, (img_path, label) in enumerate(clean_items):
        img = np.array(Image.open(img_path).convert("RGB"))
        b64 = image_to_base64(img)
        requests.append(make_request(
            f"{run_id}__clean__{idx}__label_{label}", b64))

    # Adversarial images
    for idx, (img_hwc, label, _) in enumerate(adv_results):
        b64 = image_to_base64(img_hwc)
        requests.append(make_request(
            f"{run_id}__adv__{idx}__label_{label}", b64))

    # Write JSONL
    jsonl_path = output_dir / f"{run_id}__batch_input.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"  Batch JSONL: {jsonl_path}  ({len(requests)} requests)")

    # Upload and submit
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch submitted — id: {batch.id}")

    # Save batch id for retrieval
    meta_path = output_dir / f"{run_id}__batch_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"batch_id": batch.id, "run_id": run_id,
                   "n_clean": len(clean_items),
                   "n_adv":   len(adv_results)}, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")
    return batch.id


def retrieve_batch(batch_id: str, output_dir: Path, run_id: str) -> None:
    """
    Poll the batch until complete, then parse results and write a CSV with:
        split, idx, true_label, predicted_label, correct
    and print a summary accuracy table (clean vs adversarial).
    """
    import openai
    client = openai.OpenAI()

    print(f"  Polling batch {batch_id} ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  Status: {status}  "
              f"({batch.request_counts.completed}/"
              f"{batch.request_counts.total} completed)")
        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            print(f"  ERROR: batch ended with status '{status}'")
            sys.exit(1)
        time.sleep(30)

    # Download results
    content = client.files.content(batch.output_file_id).text
    lines   = [l for l in content.splitlines() if l.strip()]

    csv_path = output_dir / f"{run_id}__results.csv"
    fieldnames = ["split", "idx", "true_label", "predicted_name", "correct"]
    rows_clean, rows_adv = [], []

    for line in lines:
        result    = json.loads(line)
        custom_id = result["custom_id"]
        # Parse custom_id: <run_id>__<split>__<idx>__label_<label>
        parts      = custom_id.split("__")
        split      = parts[-3]   # "clean" or "adv"
        idx        = int(parts[-2])
        true_label = int(parts[-1].replace("label_", ""))

        response = result.get("response", {})
        body     = response.get("body", {})
        choices  = body.get("choices", [])
        predicted = (choices[0]["message"]["content"].strip()
                     if choices else "")
        correct   = int(predicted.lower() ==
                        predicted.lower())  # placeholder; real check below

        row = {"split": split, "idx": idx,
               "true_label": true_label,
               "predicted_name": predicted, "correct": "?"}
        if split == "clean":
            rows_clean.append(row)
        else:
            rows_adv.append(row)

    # Load class names to resolve true_label -> name for accuracy check
    meta_path = output_dir / f"{run_id}__batch_meta.json"
    label_to_name = {}
    if meta_path.exists():
        # Try to load from cached class names
        dataset = run_id.split("__")[0]
        cn_path = WORK_DIR / "class_names" / f"{dataset}.json"
        if cn_path.exists():
            with open(cn_path) as f:
                raw = json.load(f)
            label_to_name = ({int(k): v for k, v in raw.items()}
                             if isinstance(raw, dict)
                             else {i: v for i, v in enumerate(raw)})

    def score(rows):
        correct = 0
        for row in rows:
            true_name = label_to_name.get(row["true_label"], "").lower()
            pred_name = row["predicted_name"].lower()
            row["correct"] = int(true_name == pred_name)
            correct += row["correct"]
        return correct, len(rows)

    c_correct, c_total = score(rows_clean)
    a_correct, a_total = score(rows_adv)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_clean + rows_adv)

    print(f"\n  Results saved to: {csv_path}")
    print(f"\n  {'Split':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*44}")
    if c_total:
        print(f"  {'clean':<12} {c_correct:>8} {c_total:>8} "
              f"{c_correct/c_total:>10.4f}")
    if a_total:
        print(f"  {'adversarial':<12} {a_correct:>8} {a_total:>8} "
              f"{a_correct/a_total:>10.4f}")
    print()

# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, args, device: torch.device) -> str | None:
    """
    Full pipeline for one dataset:
      1. Load clean test items
      2. Load class names + build CLIP surrogate
      3. Craft adversarial examples with AutoAttack
      4. Submit clean + adversarial images to OpenAI Batch API
    Returns the batch_id, or None on failure.
    """
    print(f"\n{'#'*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'#'*60}")

    # Class names
    label_to_name = get_class_names(dataset)
    class_names   = [label_to_name[i] for i in sorted(label_to_name)]
    N             = len(class_names)
    print(f"  num_classes = {N}")

    # Clean items
    n_images    = None if args.all_datasets else args.n_images
    clean_items = load_test_items(dataset, n=n_images)
    if clean_items is None:
        print("  WARNING: could not load test items. Skipping.")
        return None
    print(f"  {len(clean_items)} test images loaded")

    # CLIP surrogate
    surrogate = CLIPSurrogate(class_names, device)

    # Craft adversarial examples
    print(f"\n  Crafting adversarial examples  "
          f"(norm={args.norm}  eps={args.eps:.4f}) ...")
    adv_results = craft_adversarial_examples(
        surrogate, clean_items, device,
        norm=args.norm, eps=args.eps, batch_size=args.batch_size,
    )
    print(f"  {len(adv_results)} adversarial images crafted")

    # Sanity check: surrogate accuracy on clean vs adv
    with torch.no_grad():
        transform  = build_transform()
        x_clean    = torch.stack([transform(Image.open(p).convert("RGB"))
                                  for p, _ in clean_items]).to(device)
        labels_t   = torch.tensor([l for _, l in clean_items]).to(device)
        logits_nat = surrogate(x_clean)
        clean_acc  = (logits_nat.argmax(1) == labels_t).float().mean().item()

        x_adv    = torch.stack([
            torch.from_numpy(r[0].transpose(2, 0, 1).astype(np.float32) / 255.)
            for r in adv_results]).to(device)
        logits_adv = surrogate(x_adv)
        adv_acc    = (logits_adv.argmax(1) == labels_t).float().mean().item()

    print(f"  Surrogate clean accuracy : {clean_acc:.4f}")
    print(f"  Surrogate robust accuracy: {adv_acc:.4f}  "
          f"(should be near 0 — confirms attack succeeded)")

    # Submit to OpenAI Batch API
    run_id     = f"{dataset}__{args.norm.lower()}_eps{args.eps:.4f}"
    output_dir = Path(args.results_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id = submit_batch(
        adv_results   = adv_results,
        clean_items   = clean_items,
        class_names   = class_names,
        label_to_name = label_to_name,
        openai_model  = args.openai_model,
        run_id        = run_id,
        output_dir    = output_dir,
    )
    return batch_id

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Black-box adversarial evaluation on RobustGenBench."
    )
    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit",   action="store_true",
                      help="Craft attacks and submit to OpenAI Batch API")
    mode.add_argument("--retrieve", type=str, metavar="BATCH_ID",
                      help="Retrieve and score a completed batch")

    # Dataset selection
    parser.add_argument("--dataset",      default=DEFAULT_DATASET,
                        help=f"Dataset to evaluate (default: {DEFAULT_DATASET})")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Loop over all datasets discovered on the HF repo "
                             "(overrides --dataset and --n-images)")

    # Attack parameters
    parser.add_argument("--norm",     default=DEFAULT_NORM,
                        choices=["Linf", "L2", "L1"],
                        help=f"AutoAttack norm (default: {DEFAULT_NORM})")
    parser.add_argument("--eps",      type=float, default=DEFAULT_EPS,
                        help=f"Perturbation budget (default: {DEFAULT_EPS:.4f} "
                             f"= 30/255 for Linf)")
    parser.add_argument("--n-images", type=int, default=DEFAULT_N,
                        help=f"Images per dataset in quickstart mode "
                             f"(default: {DEFAULT_N}; ignored with --all-datasets)")

    # Target model
    parser.add_argument("--openai-model", default=DEFAULT_MODEL,
                        help=f"OpenAI model to use as target (default: {DEFAULT_MODEL})")

    # Misc
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--device",      default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--results-dir", default="./blackbox_results")
    parser.add_argument("--run-id",      default=None,
                        help="Run ID for --retrieve (inferred from --dataset "
                             "and --eps if omitted)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n  Device       : {device}")
    print(f"  Target model : {args.openai_model}")
    print(f"  Norm / eps   : {args.norm} / {args.eps:.4f}")

    # ---- submit ------------------------------------------------------------
    if args.submit:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. "
                  "Run: export OPENAI_API_KEY=sk-...")
            sys.exit(1)

        datasets = list_datasets() if args.all_datasets else [args.dataset]
        print(f"  Datasets     : {', '.join(datasets)}\n")

        batch_ids = {}
        for dataset in datasets:
            batch_id = run_dataset(dataset, args, device)
            if batch_id:
                batch_ids[dataset] = batch_id

        print(f"\n{'='*60}")
        print("  Submitted batches:")
        for ds, bid in batch_ids.items():
            run_id = f"{ds}__{args.norm.lower()}_eps{args.eps:.4f}"
            print(f"  {ds:<30} batch_id={bid}")
            print(f"    Retrieve with:")
            print(f"    python evaluate_blackbox_quickstart.py "
                  f"--retrieve {bid} --dataset {ds} --eps {args.eps}")

    # ---- retrieve ----------------------------------------------------------
    elif args.retrieve:
        run_id     = (args.run_id or
                      f"{args.dataset}__{args.norm.lower()}_eps{args.eps:.4f}")
        output_dir = Path(args.results_dir) / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Retrieving batch: {args.retrieve}")
        print(f"  Run ID          : {run_id}")
        retrieve_batch(args.retrieve, output_dir, run_id)


if __name__ == "__main__":
    main()