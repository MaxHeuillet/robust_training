#!/bin/bash
# =============================================================================
# setup_paths.sh
# Project-specific path configuration for robust_training on TAMIA.
# Sourced by job scripts AFTER execute_setup.sh.
# execute_setup.sh is left completely untouched.
# =============================================================================

CODE_DIR="$HOME/links/projects/aip-adurand/mheuill/robust_training"
SCRATCH_DIR="$HOME/links/scratch/mheuill/robust_training"

# Configs (HPO yamls, default configs) — lives next to the code
export CONFIGS_PATH="${CODE_DIR}/configs"

# Trained model state dicts (.pt files) — scratch, persists across jobs
export TRAINED_STATEDICTS_PATH="${SCRATCH_DIR}/trained_statedicts"

# Evaluation outputs (CSV results from RobustGenBench) — projects folder
export ROBUSTGENBENCH_RESULTS_PATH="${CODE_DIR}/robustgenbench_eval"

# Training + test data — re-downloaded each job into node-local tmpdir
export DATASET_PATH="${SLURM_TMPDIR}/data"

# Adversarial archives cache — tmpdir (re-downloaded each job)
export ADV_ARCHIVES_PATH="${SLURM_TMPDIR}/adv_archives"

# HuggingFace cache — tmpdir so we don't fill scratch with model weights
export HF_HOME="${SLURM_TMPDIR}/hf_cache"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_cache/datasets"

# Work dir for Ray / extracted archives
export WORK_DIR="${SLURM_TMPDIR}/work"

# Create persistent dirs (tmpdir ones are created automatically by the OS)
mkdir -p "${TRAINED_STATEDICTS_PATH}"
mkdir -p "${ROBUSTGENBENCH_RESULTS_PATH}"