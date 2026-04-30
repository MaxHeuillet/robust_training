#!/usr/bin/env python3
"""
create_sample_repo.py

Creates legolasflagstaff/RobustGenBench-sample on HuggingFace.

Sampling strategy:
  - For EVERY archive in the full dataset (all 6 datasets × all strata),
    extract exactly N_PER_CLASS=2 images per class using test/labels.csv.
  - The sample index (which filenames) is determined once per dataset from
    its clean archive, then reused across all adversarial archives for that
    dataset — so clean/adversarial pairs are matchable by filename.
  - Images are uploaded as raw PNGs (no archives), plus labels.csv and
    metadata.json so the folder is self-contained.

Run from any directory. Requires:
    pip install huggingface_hub zstandard
    huggingface-cli login   (or set HF_TOKEN env var)

Estimated sample size: 2 imgs/class × ~100 classes × ~230 archives ≈ ~500MB
"""

import csv
import io
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import zstandard
import tarfile
from huggingface_hub import HfApi, hf_hub_download

# ── Configuration ─────────────────────────────────────────────────────────────

SOURCE_REPO  = "legolasflagstaff/RobustGenBench"
SAMPLE_REPO  = "legolasflagstaff/RobustGenBench-sample"
REPO_TYPE    = "dataset"
N_PER_CLASS  = 2
SPLIT        = "test"

# All (source_path, sample_dest_folder) pairs — every archive in the full repo.
# dest_folder mirrors the source structure with clean/ prefix for root archives.
ALL_ARCHIVES = [
    # ── Clean (root level) ────────────────────────────────────────────────
    ("caltech101_processed.tar.zst",                          "clean/caltech101"),
    ("fgvc-aircraft-2013b_processed.tar.zst",                 "clean/fgvc-aircraft-2013b"),
    ("flowers-102_processed.tar.zst",                         "clean/flowers-102"),
    ("oxford-iiit-pet_processed.tar.zst",                     "clean/oxford-iiit-pet"),
    ("stanford_cars_processed.tar.zst",                       "clean/stanford_cars"),
    ("uc-merced-land-use-dataset_processed.tar.zst",          "clean/uc-merced-land-use-dataset"),
    # ── Common corruptions ────────────────────────────────────────────────
    ("adversarial/common/common_severity3/caltech101__common_severity3_processed.tar.zst",                          "adversarial/common/common_severity3/caltech101"),
    ("adversarial/common/common_severity3/fgvc-aircraft-2013b__common_severity3_processed.tar.zst",                 "adversarial/common/common_severity3/fgvc-aircraft-2013b"),
    ("adversarial/common/common_severity3/flowers-102__common_severity3_processed.tar.zst",                         "adversarial/common/common_severity3/flowers-102"),
    ("adversarial/common/common_severity3/oxford-iiit-pet__common_severity3_processed.tar.zst",                     "adversarial/common/common_severity3/oxford-iiit-pet"),
    ("adversarial/common/common_severity3/stanford_cars__common_severity3_processed.tar.zst",                       "adversarial/common/common_severity3/stanford_cars"),
    ("adversarial/common/common_severity3/uc-merced-land-use-dataset__common_severity3_processed.tar.zst",          "adversarial/common/common_severity3/uc-merced-land-use-dataset"),
    # ── Random perturbations ──────────────────────────────────────────────
    ("adversarial/random/linf_eps30_random_uniform/caltech101__random__linf_eps30_random_uniform_processed.tar.zst",                         "adversarial/random/linf_eps30_random_uniform/caltech101"),
    ("adversarial/random/linf_eps30_random_uniform/fgvc-aircraft-2013b__random__linf_eps30_random_uniform_processed.tar.zst",                "adversarial/random/linf_eps30_random_uniform/fgvc-aircraft-2013b"),
    ("adversarial/random/linf_eps30_random_uniform/flowers-102__random__linf_eps30_random_uniform_processed.tar.zst",                        "adversarial/random/linf_eps30_random_uniform/flowers-102"),
    ("adversarial/random/linf_eps30_random_uniform/oxford-iiit-pet__random__linf_eps30_random_uniform_processed.tar.zst",                    "adversarial/random/linf_eps30_random_uniform/oxford-iiit-pet"),
    ("adversarial/random/linf_eps30_random_uniform/stanford_cars__random__linf_eps30_random_uniform_processed.tar.zst",                      "adversarial/random/linf_eps30_random_uniform/stanford_cars"),
    ("adversarial/random/linf_eps30_random_uniform/uc-merced-land-use-dataset__random__linf_eps30_random_uniform_processed.tar.zst",         "adversarial/random/linf_eps30_random_uniform/uc-merced-land-use-dataset"),
    # ── zeroshot_clip_vitb16_laion2b ──────────────────────────────────────
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",          "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/caltech101__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/stanford_cars__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_clip_vith14_laion2b ──────────────────────────────────────
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",          "adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/caltech101__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/flowers-102__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/stanford_cars__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_clip_vith14_laion2b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_metaclip_vith14_fullcc2_5b ──────────────────────────────
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst",          "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps4_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/caltech101__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/flowers-102__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/stanford_cars__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_metaclip_vith14_fullcc2_5b__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_metaclip_vith14_fullcc2_5b/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_siglip2_base_patch16_224 ─────────────────────────────────
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/caltech101__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/stanford_cars__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_base_patch16_224__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_base_patch16_224/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/caltech101__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/flowers-102__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/stanford_cars__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_base_patch16_224__linf_eps30__autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_base_patch16_224/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_siglip2_so400m_patch14_384 (3 datasets only in full repo) ─
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__l1_eps300_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__l1_eps300_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__l1_eps75_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__l1_eps75_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__l2_eps2_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__l2_eps2_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__l2_eps8_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__l2_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__linf_eps4_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__linf_eps4_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__linf_eps4_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps4_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__linf_eps8_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch14_384__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch14_384__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch14_384__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch14_384/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_siglip2_so400m_patch16_naflex ────────────────────────────
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
    # ── zeroshot_siglip2_so400m_patch16_naflex_patchify ───────────────────
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l1_eps300_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps300_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",          "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst", "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",         "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l1_eps75_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l1_eps75_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l2_eps2_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps2_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",            "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",           "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__l2_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/l2_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",        "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",       "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",  "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst",    "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__linf_eps8_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps8_autoattack_standard/uc-merced-land-use-dataset"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/caltech101__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",      "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/caltech101"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/fgvc-aircraft-2013b__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/fgvc-aircraft-2013b"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/flowers-102__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",     "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/flowers-102"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/oxford-iiit-pet__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/oxford-iiit-pet"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/stanford_cars__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst",   "adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/stanford_cars"),
    ("adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/uc-merced-land-use-dataset__zeroshot_siglip2_so400m_patch16_naflex__linf_eps30_autoattack_standard_processed.tar.zst","adversarial/zeroshot_siglip2_so400m_patch16_naflex_patchify/linf_eps30_autoattack_standard/uc-merced-land-use-dataset"),
]

CLASS_NAME_FILES = [
    "class_names/caltech101.json",
    "class_names/fgvc-aircraft-2013b.json",
    "class_names/flowers-102.json",
    "class_names/oxford-iiit-pet.json",
    "class_names/stanford_cars.json",
    "class_names/uc-merced-land-use-dataset.json",
]

# ── Sampling logic ────────────────────────────────────────────────────────────

def build_sample_index(local_archive: str) -> set[str]:
    """
    Read test/labels.csv from the archive; return the set of filenames
    to keep (first N_PER_CLASS per unique class label).
    """
    counts: dict[int, int] = defaultdict(int)
    keep: set[str] = set()
    with open(local_archive, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    if member.name == f"{SPLIT}/labels.csv":
                        raw = tf.extractfile(member).read().decode("utf-8")
                        for row in csv.DictReader(raw.splitlines()):
                            label = int(row["label"])
                            if counts[label] < N_PER_CLASS:
                                keep.add(row["filename"])
                                counts[label] += 1
                        break
    print(f"    index: {len(keep)} images across {len(counts)} classes")
    return keep


def extract_sample(local_archive: str, keep: set[str], dest_dir: Path):
    """
    Stream the archive and write only sampled images + labels.csv +
    metadata.json to dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(local_archive, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    name = member.name
                    bare = Path(name).name
                    keep_file = (
                        name in ("metadata.json", f"{SPLIT}/labels.csv")
                        or (bare in keep and name.endswith(".png"))
                    )
                    if keep_file:
                        target = dest_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        fileobj = tf.extractfile(member)
                        if fileobj:
                            target.write_bytes(fileobj.read())
                            n += 1
    print(f"    extracted {n} files")

# ── README content (suppresses broken HF auto-viewer) ────────────────────────

README_MD = """\
---
viewer: false
tags:
  - adversarial-robustness
  - image-classification
  - robustness-benchmark
---

# RobustGenBench-sample

This is a **stratified sample** of the full [RobustGenBench](https://huggingface.co/datasets/legolasflagstaff/RobustGenBench) dataset (22 GB), created for NeurIPS dataset submission review.

## Structure

Each subfolder contains **2 images per class** extracted from the corresponding archive in the full dataset, plus `test/labels.csv` and `metadata.json`.

```
clean/
  caltech101/          ← 2 imgs/class × 101 classes = 202 images
  fgvc-aircraft-2013b/
  flowers-102/
  oxford-iiit-pet/
  stanford_cars/
  uc-merced-land-use-dataset/

adversarial/
  common/common_severity3/<dataset>/
  random/linf_eps30_random_uniform/<dataset>/
  zeroshot_clip_vitb16_laion2b/<threat_model>/<dataset>/
  zeroshot_clip_vith14_laion2b/<threat_model>/<dataset>/
  zeroshot_metaclip_vith14_fullcc2_5b/<threat_model>/<dataset>/
  zeroshot_siglip2_base_patch16_224/<threat_model>/<dataset>/
  zeroshot_siglip2_so400m_patch14_384/<threat_model>/<dataset>/
  zeroshot_siglip2_so400m_patch16_naflex/<threat_model>/<dataset>/
  zeroshot_siglip2_so400m_patch16_naflex_patchify/<threat_model>/<dataset>/
```

Each leaf folder has the same internal layout:
- `test/labels.csv` — `filename,label` mapping (integer class indices)
- `test/NNNNN.png` — flat-numbered PNG images (same filenames across clean & adversarial)
- `metadata.json` — split counts (clean archives only)

Class name mappings are in `class_names/<dataset>.json`.

See **[SAMPLE.md](SAMPLE.md)** for full details on sampling methodology.

## Full dataset

👉 https://huggingface.co/datasets/legolasflagstaff/RobustGenBench
"""

# ── Upload helpers ────────────────────────────────────────────────────────────

import time
from huggingface_hub import CommitOperationAdd

# How many LFS files per commit. Each file = 1 LFS request.
# 50 files/commit × ~4 commits/archive × 1 archive at a time = ~200 req/archive.
# With 35s sleep between commits we spread those 200 req over ~2 min → well under
# the 1000 req/5 min free-tier limit.
BATCH_SIZE = 50
BATCH_SLEEP = 35  # seconds between commits


def commit_files_batched(api, local_dir: Path, path_in_repo_prefix: str,
                         repo_id: str, repo_type: str, commit_prefix: str):
    """
    Upload all files under local_dir to repo in batches of BATCH_SIZE,
    sleeping BATCH_SLEEP seconds between each commit.
    Each file is committed as:  path_in_repo_prefix / relative_path_inside_local_dir
    """
    all_files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    total = len(all_files)
    batches = [all_files[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    for b_idx, batch in enumerate(batches, 1):
        ops = []
        for local_path in batch:
            rel = local_path.relative_to(local_dir)
            repo_path = f"{path_in_repo_prefix}/{rel}".replace("\\", "/")
            ops.append(CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(local_path),
            ))

        msg = f"{commit_prefix} (batch {b_idx}/{len(batches)})"
        print(f"    commit {b_idx}/{len(batches)}: {len(ops)} files...", end=" ", flush=True)

        for attempt in range(8):
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=ops,
                    commit_message=msg,
                )
                print("✓")
                break
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < 7:
                    # Parse the Retry-After seconds if present, else back off
                    wait = 300  # safe default: wait out the full 5-min window
                    import re
                    m = re.search(r"retry.after[=: ]+(\d+)", err, re.IGNORECASE)
                    if m:
                        wait = int(m.group(1)) + 5
                    print(f"\n    ⏳ 429 rate limit — waiting {wait}s (attempt {attempt+1}/7)...")
                    time.sleep(wait)
                else:
                    raise

        if b_idx < len(batches):
            time.sleep(BATCH_SLEEP)


def upload_single_file(api, data, path_in_repo: str, repo_id: str,
                       repo_type: str, commit_message: str):
    """Upload one file (bytes or path string) with retry on 429."""
    for attempt in range(6):
        try:
            api.upload_file(
                path_or_fileobj=data,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                wait = 300
                print(f"  ⏳ 429 — waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api = HfApi()

    # ── 1. Ensure repo exists ──────────────────────────────────────────────
    try:
        api.repo_info(repo_id=SAMPLE_REPO, repo_type=REPO_TYPE)
        print(f"✓ Repo '{SAMPLE_REPO}' exists.")
    except Exception:
        print(f"Creating '{SAMPLE_REPO}'...")
        api.create_repo(repo_id=SAMPLE_REPO, repo_type=REPO_TYPE, private=False)

    # ── 2. Upload README.md immediately (suppresses broken auto-viewer) ───
    print("\n── Uploading README.md ──────────────────────────────────────────")
    upload_single_file(
        api, README_MD.encode(), "README.md",
        SAMPLE_REPO, REPO_TYPE, "add README (disable viewer, document structure)",
    )
    print("  ✓ README.md")

    # ── 3. Build per-dataset sample indices from the 6 clean archives ─────
    print("\n── Building per-dataset sample indices ──────────────────────────")
    dataset_indices: dict[str, set[str]] = {}
    for src_path, dest_folder in ALL_ARCHIVES[:6]:
        dataset_name = dest_folder.replace("clean/", "")
        print(f"  {dataset_name}")
        local = hf_hub_download(repo_id=SOURCE_REPO, filename=src_path, repo_type=REPO_TYPE)
        dataset_indices[dataset_name] = build_sample_index(local)

    # ── 4. Build set of already-uploaded dest_folders (for resume) ────────
    print("\n── Checking already-uploaded archives (resume support) ──────────")
    try:
        existing_files = set(api.list_repo_files(repo_id=SAMPLE_REPO, repo_type=REPO_TYPE))
        done = {
            f.replace("/test/labels.csv", "")
            for f in existing_files
            if f.endswith("/test/labels.csv")
        }
        print(f"  {len(done)} archives already uploaded, will skip.")
    except Exception:
        existing_files = set()
        done = set()
        print("  Could not fetch file list — will upload all.")

    # ── 5. Process every archive ───────────────────────────────────────────
    print(f"\n── Processing {len(ALL_ARCHIVES)} archives ───────────────────────────────")
    with tempfile.TemporaryDirectory(prefix="rgbench_") as tmpdir:
        tmpdir = Path(tmpdir)

        for i, (src_path, dest_folder) in enumerate(ALL_ARCHIVES, 1):
            dataset_name = dest_folder.split("/")[-1]
            keep = dataset_indices.get(dataset_name)
            if keep is None:
                print(f"\n  [{i}/{len(ALL_ARCHIVES)}] ⚠ unknown dataset '{dataset_name}', skipping")
                continue

            print(f"\n  [{i}/{len(ALL_ARCHIVES)}] {dest_folder}")

            if dest_folder in done:
                print("    ✓ already uploaded, skipping.")
                continue

            local = hf_hub_download(repo_id=SOURCE_REPO, filename=src_path, repo_type=REPO_TYPE)
            extract_dir = tmpdir / str(i)
            extract_sample(local, keep, extract_dir)

            # Upload in small batched commits — never hits rate limit
            commit_files_batched(
                api,
                local_dir=extract_dir,
                path_in_repo_prefix=dest_folder,
                repo_id=SAMPLE_REPO,
                repo_type=REPO_TYPE,
                commit_prefix=f"sample: {dest_folder}",
            )
            shutil.rmtree(extract_dir)

        # ── 6. Upload class_names JSONs ────────────────────────────────────
        print("\n── Uploading class_names ─────────────────────────────────────")
        for json_path in CLASS_NAME_FILES:
            if json_path in existing_files:
                print(f"  ✓ {json_path} (already uploaded)")
                continue
            local = hf_hub_download(repo_id=SOURCE_REPO, filename=json_path, repo_type=REPO_TYPE)
            upload_single_file(
                api, local, json_path,
                SAMPLE_REPO, REPO_TYPE, f"add {json_path}",
            )
            print(f"  ✓ {json_path}")

        # ── 7. Upload SAMPLE.md ────────────────────────────────────────────
        sample_md = Path(__file__).parent / "SAMPLE.md"
        if sample_md.exists():
            upload_single_file(
                api, str(sample_md), "SAMPLE.md",
                SAMPLE_REPO, REPO_TYPE, "add SAMPLE.md",
            )
            print("  ✓ SAMPLE.md")
        else:
            print("  ⚠ SAMPLE.md not found next to script — skipping")

    print(f"\n✅ Done!  https://huggingface.co/datasets/{SAMPLE_REPO}")


if __name__ == "__main__":
    main()