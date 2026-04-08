#!/usr/bin/env bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=02:59:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/siglip2_so400m_384/%x_%j.out
#SBATCH --error=logs/siglip2_so400m_384/%x_%j.err

# ============================================================
# slurm_siglip2_so400m_384_linf30.sh
#
# Crafts adversarial perturbations for SigLIP2 SO400M patch14-384
# Linf eps=30 on caltech101, flowers-102, uc-merced-land-use-dataset
# Each dataset uses 4 GPUs (sharded), datasets processed sequentially.
# ============================================================

source ./execute_setup.sh

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
SHARD="$SCRIPT_DIR/craft_shard.py"
LOG_DIR="$SCRIPT_DIR/logs/siglip2_so400m_384"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"

TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -f "$TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$TOKEN_FILE")"
else
    echo "  WARNING: no HF token at $TOKEN_FILE"
fi

SURROGATE="siglip2_so400m_384"
SURROGATE_SLUG="zeroshot_siglip2_so400m_patch14_384"
HF_REPO="MaxHeuillet/RobustGenBench"
NORM="Linf"
EPS="30"
TM_SLUG="linf_eps30_autoattack_standard"
BATCH_SIZE=32
N_GPUS=4
N_SHARDS=$N_GPUS

DATASETS=(
    "caltech101"
    "flowers-102"
    "uc-merced-land-use-dataset"
)

PIDS=()
cleanup() {
    echo "  Interrupted — killing child processes..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

already_on_hf() {
    local DATASET="$1"
    local ARCHIVE="${DATASET}__${SURROGATE_SLUG}__${TM_SLUG}_processed.tar.zst"
    local HF_PATH="adversarial/${SURROGATE_SLUG}/${TM_SLUG}/${ARCHIVE}"
    python - << PYEOF 2>/dev/null
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    files = api.list_repo_files("$HF_REPO", repo_type="dataset")
    found = "$HF_PATH" in list(files)
    sys.exit(0 if found else 1)
except Exception:
    sys.exit(1)
PYEOF
}

echo "============================================================"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  Surrogate    : SigLIP2 SO400M patch14-384"
echo "  Threat model : $NORM eps=$EPS  ($TM_SLUG)"
echo "  GPUs         : $N_GPUS"
echo "  Datasets     : ${DATASETS[*]}"
echo "============================================================"

PENDING_DATASETS=()
for DS in "${DATASETS[@]}"; do
    if already_on_hf "$DS"; then
        echo "  ✓ $DS — already on HF, skipping"
    else
        echo "  ○ $DS — needs processing"
        PENDING_DATASETS+=("$DS")
    fi
done

if [[ ${#PENDING_DATASETS[@]} -eq 0 ]]; then
    echo "  All datasets already uploaded — nothing to do."
    exit 0
fi

TOTAL_OK=0
TOTAL_FAIL=0

for DS in "${PENDING_DATASETS[@]}"; do
    echo ""
    echo "  ── $DS ──────────────────────────────────────────────"

    PIDS=()
    for SHARD_IDX in $(seq 0 $((N_SHARDS - 1))); do
        LOG_FILE="$LOG_DIR/${DS}__shard${SHARD_IDX}__${SLURM_JOB_ID:-local}.log"

        CUDA_VISIBLE_DEVICES=$SHARD_IDX \
        python "$SHARD" \
            --dataset    "$DS" \
            --norm       "$NORM" \
            --eps        "$EPS" \
            --surrogate  "$SURROGATE" \
            --batch_size "$BATCH_SIZE" \
            --n_shards   "$N_SHARDS" \
            --shard_idx  "$SHARD_IDX" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        echo "    GPU $SHARD_IDX → shard $SHARD_IDX  (PID ${PIDS[-1]})"
    done

    echo "    Waiting for $N_SHARDS shards..."
    SHARD_FAILED=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || SHARD_FAILED=$((SHARD_FAILED + 1))
    done

    if [[ $SHARD_FAILED -gt 0 ]]; then
        echo "    ✗ $DS — $SHARD_FAILED shard(s) failed, skipping merge"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    MERGE_LOG="$LOG_DIR/${DS}__merge__${SLURM_JOB_ID:-local}.log"
    if python "$SHARD" \
        --dataset    "$DS" \
        --norm       "$NORM" \
        --eps        "$EPS" \
        --surrogate  "$SURROGATE" \
        --n_shards   "$N_SHARDS" \
        --merge \
        --upload_hf \
        > "$MERGE_LOG" 2>&1; then
        echo "    ✓ $DS — merged & uploaded"
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        echo "    ✗ $DS — merge/upload failed (see $MERGE_LOG)"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Job ${SLURM_JOB_ID:-local} complete: $NORM eps=$EPS"
echo "  Succeeded : $TOTAL_OK"
echo "  Failed    : $TOTAL_FAIL"
echo "============================================================"