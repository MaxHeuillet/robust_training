"""
LLM Vision Classification Benchmark
====================================
Evaluates LLM vision APIs on fine-grained image classification datasets.

Step 1 (current): Clean / raw image classification
Step 2 (future):  Adversarial perturbation evaluation (requires local GPU)

Supported providers: anthropic, openai, google, xai
Supported datasets:  stanford_cars, oxford-iiit-pet, caltech101, flowers-102,
                     fgvc-aircraft-2013b, uc-merced-land-use-dataset

Prerequisites:
    1. Run generate_class_names.py to create class_names/*.json mappings
    2. Extract your *_processed.tar.zst archives so data is at:
       {data_root}/{dataset_name}/test/labels.csv + *.png files
    3. Set API keys as environment variables

Usage:
    # Single run with Haiku (cheapest Claude model)
    python llm_classify.py --provider anthropic --model claude-haiku-4-5-20251001 \
                           --dataset flowers-102 --max_samples 50 \
                           --data_root ~/data

    # Run on all datasets with one model
    python llm_classify.py --provider anthropic --model claude-haiku-4-5-20251001 \
                           --all_datasets --max_samples 100 --data_root ~/data

    # Full grid: all providers × all datasets
    python llm_classify.py --run_grid --max_samples 100 --data_root ~/data

    # Resume an interrupted run
    python llm_classify.py --provider anthropic --model claude-haiku-4-5-20251001 \
                           --dataset flowers-102 --resume --data_root ~/data

    # List available models
    python llm_classify.py --list_models --data_root .
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider / model registry
# ---------------------------------------------------------------------------
PROVIDER_MODELS = {
    "anthropic": [
        "claude-haiku-4-5-20251001",      # cheapest — start here
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-4o-mini",                    # cheapest
        "gpt-4o",
        "gpt-4.1",
    ],
    "google": [
        "gemini-2.5-flash",               # cheapest
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
    ],
    "xai": [
        "grok-2-vision-latest",           # cheapest
        "grok-4",
    ],
}

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# ---------------------------------------------------------------------------
# Local dataset loading (extract from tar.zst archives)
# ---------------------------------------------------------------------------

def extract_archive(data_root: str, dataset_name: str, work_dir: str) -> Path:
    """
    Extract a {dataset_name}_processed.tar.zst archive into work_dir.
    Mirrors the move_dataset_to_tmpdir logic from the training pipeline.

    Returns the path to the extracted dataset directory.
    """
    import tarfile
    try:
        import zstandard as zstd
    except ImportError:
        logger.error("Please install zstandard: pip install zstandard")
        sys.exit(1)

    archive_path = Path(data_root) / f"{dataset_name}_processed.tar.zst"
    dest_dir = Path(work_dir) / dataset_name

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Skip if already extracted
    if (dest_dir / "test" / "labels.csv").exists():
        logger.info(f"Already extracted: {dest_dir}")
        return dest_dir

    logger.info(f"Extracting {archive_path} into {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=dest_dir)

    logger.info(f"Extraction complete: {dest_dir}")
    return dest_dir


def load_local_dataset(
    dataset_dir: Path,
    split: str,
    max_samples: Optional[int] = None,
) -> list[tuple[Path, int]]:
    """
    Load image paths and integer labels from an extracted dataset directory.

    Expects:
        {dataset_dir}/{split}/labels.csv
        {dataset_dir}/{split}/*.png

    Returns list of (image_path, int_label).
    """
    split_dir = dataset_dir / split
    csv_path = split_dir / "labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {csv_path}. "
            f"Extraction may have failed."
        )

    items = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = split_dir / row["filename"]
            label = int(row["label"])
            items.append((img_path, label))

    if max_samples is not None and max_samples < len(items):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(items), size=max_samples, replace=False)
        items = [items[i] for i in sorted(indices)]

    logger.info(f"Loaded {len(items)} samples from {dataset_dir.name}/{split}")
    return items


def load_class_names(class_names_dir: str, dataset_name: str) -> dict[int, str]:
    """
    Load the integer-to-class-name mapping from class_names/{dataset_name}.json.

    Returns dict like {0: "accordion", 1: "airplanes", ...}.
    """
    path = Path(class_names_dir) / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {path}\n"
            f"Run: python generate_class_names.py --datasets_path <your_raw_data_path>"
        )
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Image transform & encoding
# ---------------------------------------------------------------------------

def get_test_transform():
    """
    Replicate the exact test-time transform from transforms.py:
        Resize((224, 224)) → GrayscaleToRGB → ToTensor()

    Returns a transform that produces a [0,1] float tensor of shape (3,224,224).
    """
    import torchvision.transforms as T

    class GrayscaleToRGB:
        def __call__(self, img):
            if img.mode == 'L':
                img = img.convert("RGB")
            return img

    return T.Compose([
        T.Resize((224, 224)),
        GrayscaleToRGB(),
        T.ToTensor(),
    ])


def tensor_to_base64(tensor, quality: int = 95) -> str:
    """
    Convert a [0,1] float tensor (C,H,W) back to a JPEG base64 string.

    This is the key bridge: the tensor is the exact same pixel data that
    your neural network sees (pre-normalization). We convert it back to
    an image for the LLM without any quality loss from extra resizing.
    """
    import torch
    # Clamp to [0,1] and convert to uint8 PIL image
    tensor = tensor.clamp(0, 1)
    img = Image.fromarray(
        (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_and_encode_image(img_path: Path, transform=None) -> str:
    """
    Open a local image, apply the test transform, and encode as base64 JPEG.

    The LLM sees the exact same 224x224 pixel content that the neural
    network would see (before module-level normalization).
    """
    img = Image.open(img_path)
    if transform is not None:
        tensor = transform(img)
        return tensor_to_base64(tensor)
    else:
        # Fallback: just resize and encode
        img = img.convert("RGB").resize((224, 224))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_classification_prompt(class_names: list[str]) -> str:
    """
    Build a system prompt that asks the LLM to classify an image
    into exactly one of the given class names.
    """
    class_list = "\n".join(f"  - {c}" for c in class_names)
    return (
        "You are an image classification system. "
        "Given an image, you must classify it into exactly one of the following classes. "
        "Respond with ONLY the class name, nothing else. No explanation, no punctuation, "
        "no extra words. Your response must exactly match one of the class names below.\n\n"
        f"Possible classes:\n{class_list}"
    )


# ---------------------------------------------------------------------------
# Provider API wrappers (async)
# ---------------------------------------------------------------------------

async def classify_anthropic(
    img_b64: str, system_prompt: str, model: str, semaphore: asyncio.Semaphore
) -> str:
    import anthropic

    async with semaphore:
        client = anthropic.AsyncAnthropic()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=100,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {"type": "text", "text": "Classify this image."},
                        ],
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return "__ERROR__"


async def classify_openai(
    img_b64: str, system_prompt: str, model: str, semaphore: asyncio.Semaphore
) -> str:
    import openai

    async with semaphore:
        client = openai.AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": "Classify this image."},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return "__ERROR__"


async def classify_google(
    img_b64: str, system_prompt: str, model: str, semaphore: asyncio.Semaphore
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("Please install google-genai: pip install google-genai")
        return "__ERROR__"

    async with semaphore:
        try:
            client = genai.Client()
            image_bytes = base64.b64decode(img_b64)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    f"{system_prompt}\n\nClassify this image.",
                ],
                config=types.GenerateContentConfig(max_output_tokens=100),
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Google API error: {e}")
            return "__ERROR__"


async def classify_xai(
    img_b64: str, system_prompt: str, model: str, semaphore: asyncio.Semaphore
) -> str:
    import openai

    async with semaphore:
        client = openai.AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY", ""),
            base_url="https://api.x.ai/v1",
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": "Classify this image."},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"xAI API error: {e}")
            return "__ERROR__"


PROVIDER_FN = {
    "anthropic": classify_anthropic,
    "openai": classify_openai,
    "google": classify_google,
    "xai": classify_xai,
}

# ---------------------------------------------------------------------------
# Matching / scoring
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove hyphens/underscores."""
    s = s.lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def match_prediction(prediction: str, ground_truth: str, class_names: list[str]) -> bool:
    """
    Check if the LLM prediction matches the ground truth.
    Uses fuzzy matching: normalized exact match, or substring containment.
    """
    pred_norm = normalize(prediction)
    gt_norm = normalize(ground_truth)

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Ground truth contained in prediction (LLM was slightly verbose)
    if gt_norm in pred_norm:
        return True

    # Prediction contained in ground truth (LLM abbreviated)
    if pred_norm in gt_norm and len(pred_norm) > 3:
        return True

    return False


# ---------------------------------------------------------------------------
# Results data structure
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    provider: str
    model: str
    dataset: str
    split: str
    timestamp: str
    total_samples: int = 0
    correct: int = 0
    errors: int = 0
    accuracy: float = 0.0
    elapsed_seconds: float = 0.0
    per_sample: list = field(default_factory=list)

    def compute_accuracy(self):
        valid = self.total_samples - self.errors
        self.accuracy = self.correct / valid if valid > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(
    provider: str,
    model: str,
    dataset_name: str,
    data_root: str,
    class_names_dir: str,
    split: str,
    max_samples: Optional[int],
    batch_concurrency: int,
    output_dir: Path,
    work_dir: str = "/tmp/llm_classify",
    resume: bool = False,
) -> RunResult:
    """Run the full evaluation for one (provider, model, dataset) combo."""

    run_id = f"{provider}__{model}__{dataset_name}__{split}".replace("/", "_")
    results_file = output_dir / f"{run_id}.json"
    predictions_file = output_dir / f"{run_id}_predictions.jsonl"

    # ---- Extract archive & load data ----
    dataset_dir = extract_archive(data_root, dataset_name, work_dir)
    items = load_local_dataset(dataset_dir, split, max_samples)
    label_to_name = load_class_names(class_names_dir, dataset_name)
    class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]
    system_prompt = build_classification_prompt(class_names_list)

    logger.info(f"Dataset: {dataset_name}, Classes: {len(class_names_list)}, Split: {split}")

    # ---- Resume logic ----
    already_done = set()
    if resume and predictions_file.exists():
        with open(predictions_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                already_done.add(rec["index"])
        logger.info(f"Resuming: {len(already_done)} samples already completed")

    classify_fn = PROVIDER_FN[provider]
    semaphore = asyncio.Semaphore(batch_concurrency)

    result = RunResult(
        provider=provider,
        model=model,
        dataset=dataset_name,
        split=split,
        timestamp=datetime.now().isoformat(),
    )

    logger.info(f"Starting: {provider}/{model} on {dataset_name}/{split}")
    logger.info(f"  Samples: {len(items)}, Concurrency: {batch_concurrency}")
    t0 = time.time()

    # ---- Encode images ----
    logger.info("Encoding images (applying test transform: Resize(224) → RGB → ToTensor)...")
    transform = get_test_transform()
    encoded = []
    for idx, (img_path, label) in enumerate(items):
        if idx in already_done:
            encoded.append(None)
        else:
            encoded.append(load_and_encode_image(img_path, transform=transform))

    # ---- Send requests ----
    logger.info("Sending API requests...")

    async def process_one(idx: int, img_b64: Optional[str], int_label: int):
        if idx in already_done:
            return None
        gt_name = label_to_name[int_label]
        prediction = await classify_fn(img_b64, system_prompt, model, semaphore)
        is_correct = match_prediction(prediction, gt_name, class_names_list)
        return {
            "index": idx,
            "int_label": int_label,
            "ground_truth": gt_name,
            "prediction": prediction,
            "correct": is_correct,
            "error": prediction == "__ERROR__",
        }

    tasks = [
        process_one(idx, enc, label)
        for idx, ((img_path, label), enc) in enumerate(zip(items, encoded))
    ]

    # Process with progress tracking
    completed = len(already_done)
    total = len(items)

    with open(predictions_file, "a") as f_pred:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            if rec is None:
                continue
            completed += 1
            f_pred.write(json.dumps(rec) + "\n")
            f_pred.flush()

            if completed % 25 == 0 or completed == total:
                logger.info(f"  Progress: {completed}/{total}")

    # ---- Aggregate ----
    all_records = []
    with open(predictions_file, "r") as f:
        for line in f:
            all_records.append(json.loads(line))

    result.total_samples = len(all_records)
    result.correct = sum(1 for r in all_records if r["correct"])
    result.errors = sum(1 for r in all_records if r["error"])
    result.elapsed_seconds = round(time.time() - t0, 2)
    result.compute_accuracy()
    result.per_sample = all_records

    # ---- Save ----
    with open(results_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(
        f"DONE: {provider}/{model} on {dataset_name}/{split} => "
        f"Accuracy: {result.accuracy:.4f} "
        f"({result.correct}/{result.total_samples - result.errors} valid, "
        f"{result.errors} errors) in {result.elapsed_seconds}s"
    )
    return result


# ---------------------------------------------------------------------------
# Grid / multi-dataset modes
# ---------------------------------------------------------------------------

async def run_multiple(
    providers_models: list[tuple[str, str]],
    dataset_names: list[str],
    data_root: str,
    class_names_dir: str,
    split: str,
    max_samples: Optional[int],
    batch_concurrency: int,
    output_dir: Path,
    work_dir: str = "/tmp/llm_classify",
):
    """Run evaluation for multiple (provider, model, dataset) combos."""
    results = []
    for provider, model in providers_models:
        for dataset_name in dataset_names:
            try:
                r = await evaluate(
                    provider=provider,
                    model=model,
                    dataset_name=dataset_name,
                    data_root=data_root,
                    class_names_dir=class_names_dir,
                    split=split,
                    max_samples=max_samples,
                    batch_concurrency=batch_concurrency,
                    output_dir=output_dir,
                    work_dir=work_dir,
                )
                results.append(r)
            except Exception as e:
                logger.error(f"Failed: {provider}/{model} on {dataset_name}: {e}")

    # ---- Summary table ----
    if results:
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)
        print(
            f"{'Provider':<12} {'Model':<30} {'Dataset':<30} "
            f"{'Accuracy':>8} {'Correct':>8} {'Errors':>7} {'Time':>7}"
        )
        print("-" * 95)
        for r in results:
            print(
                f"{r.provider:<12} {r.model:<30} {r.dataset:<30} "
                f"{r.accuracy:>8.4f} {r.correct:>8} {r.errors:>7} "
                f"{r.elapsed_seconds:>6.0f}s"
            )

        summary_file = output_dir / "summary.json"
        summary = [
            {
                "provider": r.provider,
                "model": r.model,
                "dataset": r.dataset,
                "split": r.split,
                "accuracy": r.accuracy,
                "correct": r.correct,
                "total": r.total_samples,
                "errors": r.errors,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in results
        ]
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Vision Classification Benchmark",
    )

    # --- Mode ---
    parser.add_argument("--run_grid", action="store_true",
                        help="Run all providers (cheapest model) x all datasets")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run one provider/model on all 6 datasets")

    # --- Single-run ---
    parser.add_argument("--provider", type=str, choices=list(PROVIDER_MODELS.keys()))
    parser.add_argument("--model", type=str,
                        help="Model name (default: cheapest for provider)")
    parser.add_argument("--dataset", type=str, choices=DATASETS)

    # --- Data paths ---
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir containing *_processed.tar.zst archives")
    parser.add_argument("--class_names_dir", type=str, default="./class_names",
                        help="Dir with class_names/{dataset}.json files")
    parser.add_argument("--work_dir", type=str, default="/tmp/llm_classify",
                        help="Temp dir for extracting archives (default: /tmp/llm_classify)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "test_common"])

    # --- Shared ---
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_concurrency", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="./llm_classification_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list_models", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable provider / model combinations:\n")
        for provider, models in PROVIDER_MODELS.items():
            for m in models:
                tag = " (cheapest)" if m == models[0] else ""
                print(f"  --provider {provider:<10}  --model {m}{tag}")
        print()
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_grid:
        providers_models = [
            (p, ms[0]) for p, ms in PROVIDER_MODELS.items()
        ]
        asyncio.run(run_multiple(
            providers_models, DATASETS,
            args.data_root, args.class_names_dir, args.split,
            args.max_samples, args.batch_concurrency, output_dir,
            args.work_dir,
        ))

    elif args.all_datasets:
        if not args.provider:
            logger.error("--all_datasets requires --provider")
            sys.exit(1)
        model = args.model or PROVIDER_MODELS[args.provider][0]
        asyncio.run(run_multiple(
            [(args.provider, model)], DATASETS,
            args.data_root, args.class_names_dir, args.split,
            args.max_samples, args.batch_concurrency, output_dir,
            args.work_dir,
        ))

    else:
        if not args.provider or not args.dataset:
            logger.error("Specify --provider and --dataset, or use --run_grid / --all_datasets")
            sys.exit(1)
        model = args.model or PROVIDER_MODELS[args.provider][0]
        asyncio.run(evaluate(
            args.provider, model, args.dataset,
            args.data_root, args.class_names_dir, args.split,
            args.max_samples, args.batch_concurrency, output_dir,
            args.work_dir, args.resume,
        ))


if __name__ == "__main__":
    main()