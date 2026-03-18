#!/usr/bin/env python3
"""
fix_manifest.py — Reconstruct the manifest for test_v1 from the known batch IDs.

Run once from your project root:
    python fix_manifest.py
"""

import json
from pathlib import Path

OUTPUT_DIR   = Path("llm_classification_results")
MANIFEST_PATH = OUTPUT_DIR / "batch_manifest__flowers102__test_v1.json"

# Paste the batch IDs exactly as printed in your terminal
ENTRIES = [
    {
        "dataset":    "flowers-102",
        "key":        "google_nothink",
        "provider":   "google",
        "model":      "gemini-3-flash-preview-nothink",
        "run_name":   "flowers-102__google_nothink__test_v1",
        "batch_id":   "batches/7x4nctujhahbztwz04savm7k41dr63m161qj",
        "status":     "submitted",
        "experiment": "test_v1",
    },
    {
        "dataset":    "flowers-102",
        "key":        "google_think",
        "provider":   "google",
        "model":      "gemini-3-flash-preview-think",
        "run_name":   "flowers-102__google_think__test_v1",
        "batch_id":   "batches/05hb7nqbi74ayjn0gwt8xh8n1uccph11d1u8",
        "status":     "submitted",
        "experiment": "test_v1",
    },
    {
        "dataset":    "flowers-102",
        "key":        "anthropic",
        "provider":   "anthropic",
        "model":      "claude-haiku-4-5-20251001",
        "run_name":   "flowers-102__anthropic__test_v1",
        "batch_id":   "msgbatch_016UdagkGr2UoT42cNqC2PxH",
        "status":     "submitted",
        "experiment": "test_v1",
    },
    {
        "dataset":    "flowers-102",
        "key":        "openai",
        "provider":   "openai",
        "model":      "gpt-4o-mini",
        "run_name":   "flowers-102__openai__test_v1",
        "batch_id":   "batch_69ab551c3bc08190a5a7a63e73ca2f49",
        "status":     "submitted",
        "experiment": "test_v1",
    },
]

MANIFEST_PATH.write_text(json.dumps(ENTRIES, indent=2))
print(f"Wrote {len(ENTRIES)} entries to {MANIFEST_PATH}")
print()
print("Retrieve with:")
print(f"  python retrieve_batches.py {MANIFEST_PATH}")