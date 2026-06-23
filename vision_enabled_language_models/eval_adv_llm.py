#!/usr/bin/env python3
"""
eval_adv_llm.py — Send already-crafted adversarial images to LLM API via batch.
Reads directly from the adversarial_examples folder, no archive needed.

Usage:
    python eval_adv_llm.py --submit
    python eval_adv_llm.py --retrieve <batch_id>
"""

import argparse
import json
import asyncio
from pathlib import Path

# Reuse everything from llm_classify.py
from llm_classify import (
    run_batch_openai,
    retrieve_batch_results_openai,
    build_classification_prompt,
)

ADV_DIR          = Path("/Users/maximeheuillet/Desktop/robust_training/adversarial_examples/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps30__apgd-ce")
CLASS_NAMES_PATH = Path("/Users/maximeheuillet/data_processed/class_names/flowers-102.json")
OUTPUT_DIR       = Path("./llm_classification_results")
RUN_NAME         = "flowers-102__adv_linf30__zeroshot_clip__openai"

def load_items_from_adv_folder(adv_dir: Path, class_names_path: Path):
    raw          = json.loads(class_names_path.read_text())
    label_to_name = {int(k): v for k, v in raw.items()}
    meta         = [json.loads(l) for l in (adv_dir / "metadata.jsonl").read_text().splitlines() if l.strip()]
    # items = list of (img_path, label_idx) — same format as load_local_dataset()
    items        = [(adv_dir / rec["image_path"], rec["label_idx"]) for rec in meta]
    return items, label_to_name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submit",   action="store_true")
    p.add_argument("--retrieve", type=str, default=None, metavar="BATCH_ID")
    args = p.parse_args()

    raw           = json.loads(CLASS_NAMES_PATH.read_text())
    label_to_name = {int(k): v for k, v in raw.items()}
    class_names   = [label_to_name[i] for i in sorted(label_to_name)]
    system_prompt = build_classification_prompt(class_names)
    items, _      = load_items_from_adv_folder(ADV_DIR, CLASS_NAMES_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(items)} adversarial images")

    if args.submit:
        batch_id = asyncio.run(run_batch_openai(
            items         = items,
            label_to_name = label_to_name,
            class_names_list = class_names,
            system_prompt = system_prompt,
            model         = "gpt-4o-mini",
            output_dir    = OUTPUT_DIR,
            run_id        = RUN_NAME,
            dataset_name  = "flowers-102",
            data_root     = "",
            class_names_dir = "",
        ))
        print(f"\nBatch submitted: {batch_id}")
        print(f"Retrieve with:")
        print(f"  python eval_adv_llm.py --retrieve {batch_id}")

    elif args.retrieve:
        retrieve_batch_results_openai(
            batch_id       = args.retrieve,
            dataset_name   = "flowers-102",
            class_names_dir = str(CLASS_NAMES_PATH.parent),
            output_dir     = OUTPUT_DIR,
            run_id         = RUN_NAME,
        )

if __name__ == "__main__":
    main()