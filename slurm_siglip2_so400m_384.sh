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

source ./execute_setup.sh
set -euo pipefail

THREAT_MODEL="${1:?Usage: sbatch slurm_siglip2_so400m_384.sh <threat_model>}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
SHARD="$SCRIPT_DIR/craft_shard.py"
LOG_DIR="$SCRIPT_DIR/logs/siglip2_so400m_384"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"
TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -f "$TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$TOKEN_FILE")"
fi

SURROGATE="siglip2_so400m_384"
SURROGATE_SLUG="zeroshot_siglip2_so400m_patch14_384"
HF_REPO="MaxHeuillet/RobustGenBench"
BATCH_SIZE=16
N_GPUS=4
N_SHARDS=$N_GPUS

DATASETS=(
    "caltech101"
    # "fgvc-aircraft-2013b"
    "flowers-102"
    # "oxford-iiit-pet"
    # "stanford_cars"
    "uc-merced-land-use-dataset"
)

parse_threat_model() {
    case "$1" in
        linf4)   echo "Linf 4"    ;;
        linf8)   echo "Linf 8"    ;;
        linf30)  echo "Linf 30"   ;;
        l2_2)    echo "L2 2.0"    ;;
        l2_8)    echo "L2 8.0"    ;;
        l1_75)   echo "L1 75"     ;;
        l1_300)  echo "L1 300"    ;;
        *) echo "ERROR: unknown threat model $1" >&2; exit 1 ;;
    esac
}

threat_model_slug() {
    case "$1" in
        linf4)   echo "linf_eps4_autoattack_standard"   ;;
        linf8)   echo "linf_eps8_autoattack_standard"   ;;
        linf30)  echo "linf_eps30_autoattack_standard"  ;;
        l2_2)    echo "l2_eps2_autoattack_standard"     ;;
        l2_8)    echo "l2_eps8_autoattack_standard"     ;;
        l1_75)   echo "l1_eps75_autoattack_standard"    ;;
        l1_300)  echo "l1_eps300_autoattack_standard"   ;;
    esac
}

already_on_hf() {
    local DATASET="$1"
    local TM_SLUG="$2"
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

PIDS=()
cleanup() {
    for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
    exit 1
}
trap cleanup SIGINT SIGTERM

READ_ARGS=($(parse_threat_model "$THREAT_MODEL"))
NORM="${READ_ARGS[0]}"
EPS="${READ_ARGS[1]}"
TM_SLUG=$(threat_model_slug "$THREAT_MODEL")

echo "============================================================"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo "  Surrogate    : SigLIP2 SO400M patch14-384"
echo "  Threat model : $NORM eps=$EPS  ($TM_SLUG)"
echo "  Datasets     : ${DATASETS[*]}"
echo "============================================================"

PENDING_DATASETS=()
for DS in "${DATASETS[@]}"; do
    if already_on_hf "$DS" "$TM_SLUG"; then
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
        LOG_FILE="$LOG_DIR/${THREAT_MODEL}__${DS}__shard${SHARD_IDX}__${SLURM_JOB_ID:-local}.log"
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

    SHARD_FAILED=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || SHARD_FAILED=$((SHARD_FAILED + 1))
    done

    if [[ $SHARD_FAILED -gt 0 ]]; then
        echo "    ✗ $DS — $SHARD_FAILED shard(s) failed"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    MERGE_LOG="$LOG_DIR/${THREAT_MODEL}__${DS}__merge__${SLURM_JOB_ID:-local}.log"
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
        echo "    ✗ $DS — merge/upload failed"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Succeeded : $TOTAL_OK  |  Failed : $TOTAL_FAIL"
echo "============================================================"