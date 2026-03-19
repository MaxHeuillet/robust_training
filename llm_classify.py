"""
LLM Vision Classification Benchmark
====================================
Usage:
    # Single run
    python llm_classify.py --provider google --model gemini-3-flash-preview \
        --dataset flowers-102 --max_samples 50 --data_root ~/data_processed \
        --class_names_dir ~/data_processed/class_names --run_name my_first_test

    # Batch mode (async, cheaper — currently Anthropic only)
    python llm_classify.py --provider anthropic --model claude-haiku-4-5-20251001 \
        --dataset flowers-102 --batch --data_root ~/data_processed \
        --class_names_dir ~/data_processed/class_names

    # All datasets
    python llm_classify.py --provider google --model gemini-3-flash-preview \
        --all_datasets --max_samples 100 --data_root ~/data_processed \
        --class_names_dir ~/data_processed/class_names
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging — clean, minimal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# Silence noisy HTTP loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Provider / model registry
# ---------------------------------------------------------------------------
PROVIDER_MODELS = {
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
    ],
    "google": [
        "gemini-3-flash-preview-nothink",   # MINIMAL thinking (cheapest, fastest)
        "gemini-3-flash-preview-think",      # HIGH thinking (full reasoning)
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "xai": [
        "grok-2-vision-latest",
        "grok-4",
    ],
}

# Map virtual model names to actual API model + thinking config
GOOGLE_MODEL_CONFIG = {
    "gemini-3-flash-preview-nothink": {"model": "gemini-3-flash-preview", "thinking_level": "MINIMAL", "max_output_tokens": 100},
    "gemini-3-flash-preview-think":   {"model": "gemini-3-flash-preview", "thinking_level": "HIGH",    "max_output_tokens": 256},
    "gemini-2.5-flash": {"model": "gemini-2.5-flash", "thinking_level": None, "max_output_tokens": 100},
    "gemini-2.5-pro":   {"model": "gemini-2.5-pro",   "thinking_level": None, "max_output_tokens": 100},
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
# Dataset loading (extract from tar.zst)
# ---------------------------------------------------------------------------

def extract_archive(data_root: str, dataset_name: str, work_dir: str) -> Path:
    import tarfile
    try:
        import zstandard as zstd
    except ImportError:
        logger.error("pip install zstandard")
        sys.exit(1)

    archive_path = Path(data_root) / f"{dataset_name}_processed.tar.zst"
    dest_dir = Path(work_dir) / dataset_name

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if (dest_dir / "test" / "labels.csv").exists():
        return dest_dir

    logger.info(f"Extracting {archive_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=dest_dir)
    return dest_dir


def load_local_dataset(dataset_dir: Path, split: str, max_samples: Optional[int] = None):
    split_dir = dataset_dir / split
    csv_path = split_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")

    items = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            items.append((split_dir / row["filename"], int(row["label"])))

    if max_samples is not None and max_samples < len(items):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(items), size=max_samples, replace=False)
        items = [items[i] for i in sorted(indices)]
    return items


BASE_DATASETS = [
    "caltech101", "fgvc-aircraft-2013b", "flowers-102",
    "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset",
]

def get_base_dataset(dataset_name: str) -> str:
    for base in BASE_DATASETS:
        if dataset_name.startswith(base):
            return base
    return dataset_name

def load_class_names(class_names_dir: str, dataset_name: str) -> dict[int, str]:
    path = Path(class_names_dir) / f"{get_base_dataset(dataset_name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Class names not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Image transform & encoding
# ---------------------------------------------------------------------------

def get_test_transform():
    import torchvision.transforms as T

    class GrayscaleToRGB:
        def __call__(self, img):
            if img.mode == 'L':
                img = img.convert("RGB")
            return img

    return T.Compose([T.Resize((224, 224)), GrayscaleToRGB(), T.ToTensor()])


def tensor_to_base64(tensor, quality: int = 95) -> str:
    tensor = tensor.clamp(0, 1)
    img = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_and_encode_image(img_path: Path, transform=None) -> str:
    img = Image.open(img_path)
    if transform is not None:
        return tensor_to_base64(transform(img))
    img = img.convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_classification_prompt(class_names: list[str]) -> str:
    class_list = ", ".join(class_names)
    return (
        "Classify the image into exactly one of these classes: "
        f"{class_list}. "
        "Output ONLY the exact class name. No explanation, no punctuation, no extra words."
        "Your response must exactly match one of the class names above."
    )


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def parse_prediction(raw_prediction: str, class_names: list[str]) -> tuple[str, float]:
    if raw_prediction == "__ERROR__":
        return "__ERROR__", 0.0

    pred_norm = normalize(raw_prediction)
    norm_to_original = {normalize(cls): cls for cls in class_names}

    if pred_norm in norm_to_original:
        return norm_to_original[pred_norm], 1.0

    best_contained = None
    best_contained_len = 0
    for cls_norm, cls_orig in norm_to_original.items():
        if cls_norm in pred_norm and len(cls_norm) > best_contained_len:
            best_contained = cls_orig
            best_contained_len = len(cls_norm)
    if best_contained and best_contained_len > 3:
        return best_contained, 0.95

    for cls_norm, cls_orig in norm_to_original.items():
        if pred_norm in cls_norm and len(pred_norm) > 3:
            return cls_orig, 0.9

    best_match = None
    best_score = 0.0
    for cls_norm, cls_orig in norm_to_original.items():
        score = SequenceMatcher(None, pred_norm, cls_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = cls_orig

    if best_score >= 0.6:
        return best_match, best_score

    return raw_prediction, 0.0


# ---------------------------------------------------------------------------
# Provider API wrappers (async) with retry
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5


async def _retry_on_rate_limit(fn, *args, retries=MAX_RETRIES):
    for attempt in range(retries + 1):
        result = await fn(*args)
        if result != "__RATE_LIMITED__":
            return result
        if attempt < retries:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.info(f"  Rate limited, retrying in {delay}s...")
            await asyncio.sleep(delay)
    return "__ERROR__"


async def classify_anthropic(img_b64, system_prompt, model, semaphore):
    import anthropic
    async with semaphore:
        client = anthropic.AsyncAnthropic()
        try:
            response = await client.messages.create(
                model=model, max_tokens=100, system=system_prompt,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": "Classify this image."},
                ]}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            return "__RATE_LIMITED__"
        except Exception as e:
            logger.warning(f"  Anthropic error: {e}")
            return "__ERROR__"


async def classify_openai(img_b64, system_prompt, model, semaphore):
    import openai
    async with semaphore:
        client = openai.AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model=model, max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}},
                        {"type": "text", "text": "Classify this image."},
                    ]},
                ],
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            return "__RATE_LIMITED__"
        except Exception as e:
            logger.warning(f"  OpenAI error: {e}")
            return "__ERROR__"


async def classify_google(img_b64, system_prompt, model, semaphore):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("pip install google-genai")
        return "__ERROR__"

    gcfg = GOOGLE_MODEL_CONFIG.get(model, {"model": model, "thinking_level": None, "max_output_tokens": 100})
    actual_model = gcfg["model"]
    thinking_level = gcfg["thinking_level"]
    max_tokens = gcfg["max_output_tokens"]

    async with semaphore:
        try:
            client = genai.Client()
            image_bytes = base64.b64decode(img_b64)

            gen_config_kwargs = {"max_output_tokens": max_tokens}
            if thinking_level is not None:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level,
                )

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=actual_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    f"{system_prompt}\n\nClassify this image.",
                ],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                return "__RATE_LIMITED__"
            logger.warning(f"  Google error: {e}")
            return "__ERROR__"


async def classify_xai(img_b64, system_prompt, model, semaphore):
    import openai
    async with semaphore:
        client = openai.AsyncOpenAI(api_key=os.environ.get("XAI_API_KEY", ""), base_url="https://api.x.ai/v1")
        try:
            response = await client.chat.completions.create(
                model=model, max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}},
                        {"type": "text", "text": "Classify this image."},
                    ]},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                return "__RATE_LIMITED__"
            logger.warning(f"  xAI error: {e}")
            return "__ERROR__"


PROVIDER_FN = {
    "anthropic": classify_anthropic,
    "openai": classify_openai,
    "google": classify_google,
    "xai": classify_xai,
}

# ---------------------------------------------------------------------------
# Batch API (Anthropic)
# ---------------------------------------------------------------------------

async def run_batch_anthropic(
    items: list[tuple[int, Path, int]],  # (orig_idx, path, label)
    label_to_name: dict[int, str],
    class_names_list: list[str],
    system_prompt: str,
    model: str,
    output_dir: Path,
    run_id: str,
    dataset_name: str = "",
    data_root: str = "",
    class_names_dir: str = "",
):
    import anthropic

    logger.info("Preparing Anthropic batch request...")
    transform = get_test_transform()
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    requests = []
    for orig_idx, img_path, int_label in items:
        img_b64 = load_and_encode_image(img_path, transform=transform)
        requests.append({
            "custom_id": f"img-{orig_idx:05d}-label-{int_label}",
            "params": {
                "model": model,
                "max_tokens": 100,
                "system": system_prompt,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": "Classify this image."},
                ]}],
            },
        })

    batch_input_file = run_dir / "batch_input.jsonl"
    with open(batch_input_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Wrote {len(requests)} requests to {batch_input_file}")

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Batch submitted: id={batch.id}")
    logger.info(f"Status: {batch.processing_status}")

    retrieve_cmd = (
        f"python llm_classify.py --batch_retrieve {batch.id} "
        f"--dataset {dataset_name} --data_root {data_root} "
        f"--class_names_dir {class_names_dir} "
        f"--output_dir {output_dir} --run_name {run_id}"
    )
    logger.info(f"To retrieve: {retrieve_cmd}")

    meta = {
        "batch_id": batch.id,
        "run_id": run_id,
        "model": model,
        "num_requests": len(requests),
        "submitted_at": datetime.now().isoformat(),
        "retrieve_cmd": retrieve_cmd,
    }
    with open(run_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return batch.id


def retrieve_batch_results(
    batch_id: str,
    dataset_name: str,
    class_names_dir: str,
    output_dir: Path,
    run_id: str,
):
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    logger.info(f"Batch {batch_id}: {batch.processing_status}")

    if batch.processing_status != "ended":
        logger.info(f"Batch not yet complete.")
        logger.info(f"  Counts: {batch.request_counts}")
        return

    label_to_name = load_class_names(class_names_dir, dataset_name)
    class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = run_dir / "predictions.jsonl"
    records = []

    for result in client.messages.batches.results(batch_id):
        parts = result.custom_id.split("-")
        idx = int(parts[1])
        int_label = int(parts[3])
        gt_name = label_to_name[int_label]

        if result.result.type == "succeeded":
            raw_pred = result.result.message.content[0].text.strip()
            parsed, conf = parse_prediction(raw_pred, class_names_list)
            is_correct = (normalize(parsed) == normalize(gt_name))
            records.append({
                "index": idx, "int_label": int_label, "ground_truth": gt_name,
                "raw_prediction": raw_pred, "parsed_prediction": parsed,
                "confidence": conf, "correct": is_correct, "error": False,
            })
        else:
            records.append({
                "index": idx, "int_label": int_label, "ground_truth": gt_name,
                "raw_prediction": "", "parsed_prediction": "",
                "confidence": 0.0, "correct": False, "error": True,
            })

    with open(predictions_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    correct = sum(1 for r in records if r["correct"])
    errors = sum(1 for r in records if r["error"])
    valid = len(records) - errors
    acc = correct / valid if valid > 0 else 0
    logger.info(f"Results: {correct}/{valid} correct ({acc:.4f}), {errors} errors")
    logger.info(f"Saved to {predictions_file}")


# ---------------------------------------------------------------------------
# Batch API (Google)
# ---------------------------------------------------------------------------

async def run_batch_google(
    items: list[tuple[int, Path, int]],  # (orig_idx, path, label)
    label_to_name: dict,
    class_names_list: list,
    system_prompt: str,
    model: str,
    output_dir: Path,
    run_id: str,
    dataset_name: str = "",
    data_root: str = "",
    class_names_dir: str = "",
):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("pip install google-genai")
        return None

    gcfg = GOOGLE_MODEL_CONFIG.get(model, {"model": model, "thinking_level": None, "max_output_tokens": 100})
    actual_model = gcfg["model"]
    thinking_level = gcfg["thinking_level"]
    max_tokens = gcfg["max_output_tokens"]

    logger.info("Preparing Google batch request (file-upload mode)...")
    transform = get_test_transform()
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    req_config: dict = {"max_output_tokens": max_tokens}
    if thinking_level is not None:
        req_config["thinking_config"] = {"thinking_budget": -1 if thinking_level == "HIGH" else 0}

    jsonl_path = run_dir / "batch_input.jsonl"
    with open(jsonl_path, "w") as f:
        for i, (orig_idx, img_path, int_label) in enumerate(items):
            img_b64 = load_and_encode_image(img_path, transform=transform)
            record = {
                "key": f"img-{orig_idx:05d}-label-{int_label}",
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                            {"text": f"{system_prompt}\n\nClassify this image."},
                        ],
                    }],
                    "generation_config": req_config,
                },
            }
            f.write(json.dumps(record) + "\n")
            if (i + 1) % 100 == 0:
                logger.info(f"  Encoded {i + 1}/{len(items)} images...")

    logger.info(f"Wrote {len(items)} requests to {jsonl_path}")

    client = genai.Client()
    logger.info("Uploading JSONL to Files API...")
    uploaded_file = await asyncio.to_thread(
        client.files.upload,
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"{run_id}_batch_input",
            mime_type="application/jsonl",
        ),
    )
    logger.info(f"Uploaded file: {uploaded_file.name}")

    batch_job = await asyncio.to_thread(
        client.batches.create,
        model=actual_model,
        src=uploaded_file.name,
        config={"display_name": run_id},
    )

    logger.info(f"Batch submitted: {batch_job.name}")
    logger.info(f"Status: {batch_job.state}")

    retrieve_cmd = (
        f"python llm_classify.py --batch_retrieve {batch_job.name} "
        f"--provider google --dataset {dataset_name} "
        f"--data_root {data_root} --class_names_dir {class_names_dir} "
        f"--output_dir {output_dir} --run_name {run_id}"
    )
    logger.info(f"To retrieve: {retrieve_cmd}")

    meta = {
        "batch_name": batch_job.name,
        "uploaded_file": uploaded_file.name,
        "provider": "google",
        "run_id": run_id,
        "model": model,
        "actual_model": actual_model,
        "num_requests": len(items),
        "retrieve_cmd": retrieve_cmd,
    }
    with open(run_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return batch_job.name


def retrieve_batch_results_google(
    batch_name: str,
    dataset_name: str,
    class_names_dir: str,
    output_dir: Path,
    run_id: str,
):
    from google import genai
    from llm_classify import load_class_names, parse_prediction, normalize

    client = genai.Client()
    job = client.batches.get(name=batch_name)
    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    logger.info(f"Batch {batch_name}: {job.state.name}")

    if job.state.name not in completed_states:
        logger.info("Not yet complete. Check back later.")
        return

    if job.state.name != "JOB_STATE_SUCCEEDED":
        logger.error(f"Batch job did not succeed: {job.state.name}")
        return

    label_to_name = load_class_names(class_names_dir, dataset_name)
    class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = run_dir / "predictions.jsonl"

    result_file_name = job.dest.file_name
    logger.info(f"Downloading result file: {result_file_name}")
    file_content_bytes = client.files.download(file=result_file_name)
    result_lines = file_content_bytes.decode("utf-8").strip().splitlines()
    logger.info(f"Downloaded {len(result_lines)} result lines")

    records = []
    for line in result_lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
            key = result["key"]
            parts = key.split("-")
            idx = int(parts[1])
            int_label = int(parts[3])
            gt_name = label_to_name[int_label]

            response = result.get("response", {})
            candidates = response.get("candidates", [])
            if candidates:
                raw_pred = candidates[0]["content"]["parts"][0]["text"].strip()
                parsed, conf = parse_prediction(raw_pred, class_names_list)
                is_correct = normalize(parsed) == normalize(gt_name)
                records.append({
                    "index": idx, "int_label": int_label, "ground_truth": gt_name,
                    "raw_prediction": raw_pred, "parsed_prediction": parsed,
                    "confidence": conf, "correct": is_correct, "error": False,
                })
            else:
                records.append({
                    "index": idx, "int_label": int_label, "ground_truth": gt_name,
                    "raw_prediction": "", "parsed_prediction": "",
                    "confidence": 0.0, "correct": False, "error": True,
                })
        except Exception as e:
            logger.warning(f"Failed to parse result line: {e}")
            records.append({
                "index": -1, "int_label": -1, "ground_truth": "",
                "raw_prediction": str(e), "parsed_prediction": "",
                "confidence": 0.0, "correct": False, "error": True,
            })

    with open(predictions_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    correct = sum(1 for r in records if r["correct"])
    errors  = sum(1 for r in records if r["error"])
    valid   = len(records) - errors
    acc     = correct / valid if valid > 0 else 0
    logger.info(f"Results: {correct}/{valid} correct ({acc:.4f}), {errors} errors")
    logger.info(f"Saved to {predictions_file}")


# ---------------------------------------------------------------------------
# Batch API (OpenAI)
# ---------------------------------------------------------------------------

async def run_batch_openai(
    items: list[tuple[int, Path, int]],  # (orig_idx, path, label)
    label_to_name: dict[int, str],
    class_names_list: list[str],
    system_prompt: str,
    model: str,
    output_dir: Path,
    run_id: str,
    dataset_name: str = "",
    data_root: str = "",
    class_names_dir: str = "",
):
    import openai

    logger.info("Preparing OpenAI batch request...")
    transform = get_test_transform()
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_input_file = run_dir / "batch_input.jsonl"
    with open(batch_input_file, "w") as f:
        for i, (orig_idx, img_path, int_label) in enumerate(items):
            img_b64 = load_and_encode_image(img_path, transform=transform)
            request = {
                "custom_id": f"img-{orig_idx:05d}-label-{int_label}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}},
                            {"type": "text", "text": "Classify this image."},
                        ]},
                    ],
                },
            }
            f.write(json.dumps(request) + "\n")
            if (i + 1) % 100 == 0:
                logger.info(f"  Encoded {i + 1}/{len(items)} images...")

    logger.info(f"Wrote {len(items)} requests to {batch_input_file}")

    client = openai.OpenAI()
    uploaded_file = client.files.create(
        file=open(batch_input_file, "rb"),
        purpose="batch",
    )
    logger.info(f"Uploaded file: {uploaded_file.id}")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    logger.info(f"Batch submitted: id={batch.id}")
    logger.info(f"Status: {batch.status}")

    retrieve_cmd = (
        f"python llm_classify.py --batch_retrieve {batch.id} "
        f"--provider openai --dataset {dataset_name} --data_root {data_root} "
        f"--class_names_dir {class_names_dir} "
        f"--output_dir {output_dir} --run_name {run_id}"
    )
    logger.info(f"To retrieve: {retrieve_cmd}")

    meta = {
        "batch_id": batch.id,
        "file_id": uploaded_file.id,
        "provider": "openai",
        "run_id": run_id,
        "model": model,
        "num_requests": len(items),
        "submitted_at": datetime.now().isoformat(),
        "retrieve_cmd": retrieve_cmd,
    }
    with open(run_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return batch.id


def retrieve_batch_results_openai(
    batch_id: str,
    dataset_name: str,
    class_names_dir: str,
    output_dir: Path,
    run_id: str,
):
    import openai

    client = openai.OpenAI()
    batch = client.batches.retrieve(batch_id)
    logger.info(f"Batch {batch_id}: {batch.status}")

    if batch.status != "completed":
        logger.info(f"Not yet complete. Status: {batch.status}")
        if batch.request_counts:
            logger.info(f"  Counts: completed={batch.request_counts.completed}, "
                        f"failed={batch.request_counts.failed}, total={batch.request_counts.total}")
        return

    label_to_name = load_class_names(class_names_dir, dataset_name)
    class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]

    result_content = client.files.content(batch.output_file_id).content
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = run_dir / "predictions.jsonl"
    records = []

    for line in result_content.decode("utf-8").strip().split("\n"):
        result = json.loads(line)
        custom_id = result["custom_id"]
        parts = custom_id.split("-")
        idx = int(parts[1])
        int_label = int(parts[3])
        gt_name = label_to_name[int_label]

        if result.get("error"):
            records.append({
                "index": idx, "int_label": int_label, "ground_truth": gt_name,
                "raw_prediction": str(result["error"]), "parsed_prediction": "",
                "confidence": 0.0, "correct": False, "error": True,
            })
        else:
            try:
                raw_pred = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                parsed, conf = parse_prediction(raw_pred, class_names_list)
                is_correct = (normalize(parsed) == normalize(gt_name))
                records.append({
                    "index": idx, "int_label": int_label, "ground_truth": gt_name,
                    "raw_prediction": raw_pred, "parsed_prediction": parsed,
                    "confidence": conf, "correct": is_correct, "error": False,
                })
            except Exception as e:
                records.append({
                    "index": idx, "int_label": int_label, "ground_truth": gt_name,
                    "raw_prediction": str(e), "parsed_prediction": "",
                    "confidence": 0.0, "correct": False, "error": True,
                })

    with open(predictions_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    correct = sum(1 for r in records if r["correct"])
    errors = sum(1 for r in records if r["error"])
    valid = len(records) - errors
    acc = correct / valid if valid > 0 else 0
    logger.info(f"Results: {correct}/{valid} correct ({acc:.4f}), {errors} errors")
    logger.info(f"Saved to {predictions_file}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    provider: str
    model: str
    dataset: str
    split: str
    run_name: str
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
    run_name: str,
    work_dir: str = "/tmp/llm_classify",
    overwrite: bool = False,
) -> RunResult:

    run_id = run_name or f"{provider}__{model}__{dataset_name}__{split}".replace("/", "_")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"
    predictions_file = run_dir / "predictions.jsonl"

    dataset_dir = extract_archive(data_root, dataset_name, work_dir)
    items = load_local_dataset(dataset_dir, split, max_samples)
    label_to_name = load_class_names(class_names_dir, dataset_name)
    class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]
    system_prompt = build_classification_prompt(class_names_list)

    already_done = {}
    if predictions_file.exists():
        with open(predictions_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                already_done[rec["index"]] = rec

        if overwrite:
            logger.info(f"Overwriting {len(already_done)} previous results")
            already_done = {}
            predictions_file.unlink()
            if results_file.exists():
                results_file.unlink()
        else:
            successful = {idx: rec for idx, rec in already_done.items() if not rec["error"]}
            errors_prev = len(already_done) - len(successful)
            already_done = successful

            target_indices = set(range(len(items)))
            done_indices = set(already_done.keys()) & target_indices
            remaining = len(target_indices) - len(done_indices)

            if remaining == 0:
                logger.info(f"All {len(items)} samples already completed.")
                all_records = list(already_done.values())
                result = RunResult(
                    provider=provider, model=model, dataset=dataset_name,
                    split=split, run_name=run_id, timestamp=datetime.now().isoformat(),
                )
                result.total_samples = len(all_records)
                result.correct = sum(1 for r in all_records if r["correct"])
                result.errors = sum(1 for r in all_records if r["error"])
                result.compute_accuracy()
                valid = result.total_samples - result.errors
                print(f"\n  Existing results: {result.accuracy:.4f}  ({result.correct}/{valid})\n")
                return result

            if done_indices:
                logger.info(f"Resuming: {len(done_indices)} done, {remaining} remaining"
                            + (f", {errors_prev} previous errors will be retried" if errors_prev else ""))
                with open(predictions_file, "w") as f:
                    for rec in already_done.values():
                        f.write(json.dumps(rec) + "\n")

    done_indices = set(already_done.keys())
    classify_fn = PROVIDER_FN[provider]
    semaphore = asyncio.Semaphore(batch_concurrency)

    remaining = len(items) - len(done_indices)
    print(f"\n{'='*60}")
    print(f"  Run:         {run_id}")
    print(f"  Provider:    {provider}/{model}")
    print(f"  Dataset:     {dataset_name}/{split} ({len(items)} samples)")
    print(f"  Classes:     {len(class_names_list)}")
    print(f"  Remaining:   {remaining} to classify")
    print(f"  Concurrency: {batch_concurrency}")
    print(f"  Output:      {run_dir}")
    print(f"{'='*60}\n")

    result = RunResult(
        provider=provider, model=model, dataset=dataset_name,
        split=split, run_name=run_id, timestamp=datetime.now().isoformat(),
    )

    t0 = time.time()
    logger.info("Encoding images...")
    transform = get_test_transform()
    encoded = []
    for idx, (img_path, label) in enumerate(items):
        encoded.append(None if idx in done_indices else load_and_encode_image(img_path, transform=transform))

    logger.info("Classifying...")

    async def process_one(idx, img_b64, int_label):
        if idx in done_indices:
            return None
        gt_name = label_to_name[int_label]
        raw_pred = await _retry_on_rate_limit(classify_fn, img_b64, system_prompt, model, semaphore)
        parsed, conf = parse_prediction(raw_pred, class_names_list)
        is_correct = (normalize(parsed) == normalize(gt_name))
        return {
            "index": idx, "int_label": int_label, "ground_truth": gt_name,
            "raw_prediction": raw_pred, "parsed_prediction": parsed,
            "confidence": round(conf, 3), "correct": is_correct,
            "error": raw_pred == "__ERROR__",
        }

    tasks = [
        process_one(idx, enc, label)
        for idx, ((_, label), enc) in enumerate(zip(items, encoded))
    ]

    completed = len(done_indices)
    total = len(items)
    correct_so_far = sum(1 for r in already_done.values() if r.get("correct", False))

    with open(predictions_file, "a") as f_pred:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            if rec is None:
                continue
            completed += 1
            if rec["correct"]:
                correct_so_far += 1
            f_pred.write(json.dumps(rec) + "\n")
            f_pred.flush()

            if completed % 10 == 0 or completed == total:
                valid = completed - sum(1 for _ in open(predictions_file) if '"error": true' in _)
                acc = correct_so_far / completed if completed > 0 else 0
                logger.info(f"  [{completed:>4}/{total}]  running acc: {acc:.3f}")

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

    with open(results_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    valid = result.total_samples - result.errors
    print(f"\n{'='*60}")
    print(f"  DONE: {run_id}")
    print(f"  Accuracy:  {result.accuracy:.4f}  ({result.correct}/{valid})")
    print(f"  Errors:    {result.errors}")
    print(f"  Time:      {result.elapsed_seconds}s")
    print(f"  Saved:     {run_dir}")
    print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# Grid / multi-dataset
# ---------------------------------------------------------------------------

async def run_multiple(
    providers_models, dataset_names, data_root, class_names_dir,
    split, max_samples, batch_concurrency, output_dir, run_name, work_dir,
):
    results = []
    for provider, model in providers_models:
        for dataset_name in dataset_names:
            rn = f"{run_name}__{dataset_name}" if run_name else ""
            try:
                r = await evaluate(
                    provider, model, dataset_name, data_root, class_names_dir,
                    split, max_samples, batch_concurrency, output_dir, rn, work_dir,
                )
                results.append(r)
            except Exception as e:
                logger.error(f"Failed {provider}/{model} on {dataset_name}: {e}")

    if results:
        print(f"\n{'='*95}")
        print("SUMMARY")
        print(f"{'='*95}")
        print(f"{'Dataset':<30} {'Accuracy':>10} {'Correct':>10} {'Errors':>8} {'Time':>8}")
        print(f"{'-'*95}")
        for r in results:
            print(f"{r.dataset:<30} {r.accuracy:>10.4f} {r.correct:>10} {r.errors:>8} {r.elapsed_seconds:>7.0f}s")

        with open(output_dir / "summary.json", "w") as f:
            json.dump([{k: v for k, v in asdict(r).items() if k != "per_sample"} for r in results], f, indent=2)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM Vision Classification Benchmark")

    p.add_argument("--run_grid", action="store_true")
    p.add_argument("--all_datasets", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--batch_retrieve", type=str, metavar="BATCH_ID")

    p.add_argument("--provider", type=str, choices=list(PROVIDER_MODELS.keys()))
    p.add_argument("--model", type=str)
    p.add_argument("--dataset", type=str)

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--class_names_dir", type=str, default="./class_names")
    p.add_argument("--work_dir", type=str, default="/tmp/llm_classify")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test", "test_common"])

    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated list of dataset indices to process (for complement batches)")
    p.add_argument("--batch_concurrency", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="./llm_classification_results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list_models", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        for provider, models in PROVIDER_MODELS.items():
            for m in models:
                print(f"  --provider {provider:<10}  --model {m}")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Batch retrieve mode ----
    if args.batch_retrieve:
        if not args.dataset:
            logger.error("--batch_retrieve requires --dataset")
            sys.exit(1)
        run_name = args.run_name or args.batch_retrieve
        provider = args.provider or ""

        if provider == "google" or args.batch_retrieve.startswith("batches/"):
            retrieve_batch_results_google(
                args.batch_retrieve, args.dataset, args.class_names_dir,
                output_dir, run_name,
            )
        elif provider == "openai" or args.batch_retrieve.startswith("batch_"):
            retrieve_batch_results_openai(
                args.batch_retrieve, args.dataset, args.class_names_dir,
                output_dir, run_name,
            )
        else:
            retrieve_batch_results(
                args.batch_retrieve, args.dataset, args.class_names_dir,
                output_dir, run_name,
            )
        return

    # ---- Batch submit mode ----
    if args.batch:
        if not args.provider or not args.dataset:
            logger.error("--batch requires --provider and --dataset")
            sys.exit(1)
        if args.provider not in ("anthropic", "google", "openai"):
            logger.error("Batch mode supports: anthropic, google, openai")
            sys.exit(1)
        model = args.model or PROVIDER_MODELS[args.provider][0]
        run_id = args.run_name or f"batch__{model}__{args.dataset}".replace("/", "_")

        dataset_dir = extract_archive(args.data_root, args.dataset, args.work_dir)
        raw_items = load_local_dataset(dataset_dir, args.split, args.max_samples)

        # Build (orig_idx, path, label) triples — orig_idx is the dataset index
        # that will appear in custom_id/key, enabling correct merge after retrieval
        if args.indices:
            idx_set = set(int(i) for i in args.indices.split(","))
            items = [(i, path, label) for i, (path, label) in enumerate(raw_items) if i in idx_set]
            logger.info(f"Filtered to {len(items)} items from --indices (indices {min(idx_set)}-{max(idx_set)})")
        else:
            items = [(i, path, label) for i, (path, label) in enumerate(raw_items)]

        label_to_name = load_class_names(args.class_names_dir, args.dataset)
        class_names_list = [label_to_name[i] for i in sorted(label_to_name.keys())]
        system_prompt = build_classification_prompt(class_names_list)

        if args.provider == "anthropic":
            asyncio.run(run_batch_anthropic(
                items, label_to_name, class_names_list, system_prompt,
                model, output_dir, run_id,
                dataset_name=args.dataset, data_root=args.data_root,
                class_names_dir=args.class_names_dir,
            ))
        elif args.provider == "google":
            asyncio.run(run_batch_google(
                items, label_to_name, class_names_list, system_prompt,
                model, output_dir, run_id,
                dataset_name=args.dataset, data_root=args.data_root,
                class_names_dir=args.class_names_dir,
            ))
        elif args.provider == "openai":
            asyncio.run(run_batch_openai(
                items, label_to_name, class_names_list, system_prompt,
                model, output_dir, run_id,
                dataset_name=args.dataset, data_root=args.data_root,
                class_names_dir=args.class_names_dir,
            ))
        return

    # ---- Grid mode ----
    if args.run_grid:
        if not args.provider:
            logger.error("--run_grid requires --provider")
            sys.exit(1)
        model = args.model or PROVIDER_MODELS[args.provider][0]
        asyncio.run(run_multiple(
            [(args.provider, model)], DATASETS,
            args.data_root, args.class_names_dir, args.split,
            args.max_samples, args.batch_concurrency, output_dir,
            args.run_name, args.work_dir,
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
            args.run_name, args.work_dir,
        ))

    else:
        if not args.provider or not args.dataset:
            logger.error("Specify --provider and --dataset")
            sys.exit(1)
        model = args.model or PROVIDER_MODELS[args.provider][0]
        run_name = args.run_name or f"{args.provider}__{model}__{args.dataset}__{args.split}".replace("/", "_")
        asyncio.run(evaluate(
            args.provider, model, args.dataset,
            args.data_root, args.class_names_dir, args.split,
            args.max_samples, args.batch_concurrency, output_dir,
            run_name, args.work_dir, args.overwrite,
        ))


if __name__ == "__main__":
    main()