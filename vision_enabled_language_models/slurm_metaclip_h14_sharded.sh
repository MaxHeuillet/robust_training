#!/usr/bin/env bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=11:59:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/metaclip_h14/%x_%j.out
#SBATCH --error=logs/metaclip_h14/%x_%j.err

# ============================================================
# slurm_metaclip_h14_sharded.sh
#
# Processes all 6 datasets for ONE threat model.
# Each dataset uses ALL 4 GPUs (4 shards), datasets processed
# sequentially. After all shards finish, merge & upload.
#
# Submit one job per threat model:
#   sbatch -J linf8  slurm_metaclip_h14_sharded.sh linf8
#   sbatch -J linf30 slurm_metaclip_h14_sharded.sh linf30
#   sbatch -J l2_2   slurm_metaclip_h14_sharded.sh l2_2
#   sbatch -J l2_8   slurm_metaclip_h14_sharded.sh l2_8
#   sbatch -J l1_75  slurm_metaclip_h14_sharded.sh l1_75
#   sbatch -J l1_300 slurm_metaclip_h14_sharded.sh l1_300
#
# Or submit all 6 at once:
#   bash submit_all_metaclip_sharded.sh
# ============================================================

# ── Environment setup ─────────────────────────────────────────
source ./execute_setup.sh

set -euo pipefail

THREAT_MODEL="${1:?Usage: sbatch slurm_metaclip_h14_sharded.sh <threat_model>}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
CRAFT="$SCRIPT_DIR/craft_adversarial.py"
SHARD="$SCRIPT_DIR/craft_shard.py"
LOG_DIR="$SCRIPT_DIR/logs/metaclip_h14"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"

# ── HF token ──────────────────────────────────────────────────
TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -f "$TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$TOKEN_FILE")"
else
    echo "  WARNING: no HF token at $TOKEN_FILE — run: huggingface-cli login"
fi

SURROGATE="metaclip_h14"
SURROGATE_SLUG="zeroshot_metaclip_vith14_fullcc2_5b"
HF_REPO="MaxHeuillet/RobustGenBench"
BATCH_SIZE=64
N_GPUS=4
N_SHARDS=$N_GPUS

DATASETS=(
    "fgvc-aircraft-2013b"
    "flowers-102"
    "oxford-iiit-pet"
    "stanford_cars"
    "uc-merced-land-use-dataset"
    "caltech101"
)

# ── Graceful cleanup ──────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    echo "  Interrupted — killing child processes..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── Parse threat model ────────────────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────
READ_ARGS=($(parse_threat_model "$THREAT_MODEL"))
NORM="${READ_ARGS[0]}"
EPS="${READ_ARGS[1]}"
TM_SLUG=$(threat_model_slug "$THREAT_MODEL")

echo "============================================================"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  Surrogate    : MetaCLIP ViT-H/14 fullcc2.5b"
echo "  Threat model : $NORM eps=$EPS  ($TM_SLUG)"
echo "  Mode         : sharded ($N_SHARDS shards per dataset)"
echo "  GPUs         : $N_GPUS"
echo "  Datasets     : ${DATASETS[*]}"
echo "============================================================"

# ── Ensure data is downloaded ─────────────────────────────────
echo ""
echo "  Ensuring datasets are downloaded..."
python "$CRAFT" \
    --surrogate clip --norm Linf --eps 8 --max_samples 1 \
    2>&1 | grep -E "Download|already present|Extracting|ERROR" || true
echo "  Data ready."

# ── Check which datasets need processing ──────────────────────
echo ""
echo "  Checking HuggingFace for existing uploads..."
PENDING_DATASETS=()
for DS in "${DATASETS[@]}"; do
    if already_on_hf "$DS" "$TM_SLUG"; then
        echo "    ✓ $DS — already on HF, skipping"
    else
        echo "    ○ $DS — needs processing"
        PENDING_DATASETS+=("$DS")
    fi
done

if [[ ${#PENDING_DATASETS[@]} -eq 0 ]]; then
    echo ""
    echo "  All datasets already uploaded — nothing to do."
    exit 0
fi

# ── Process each dataset with all GPUs (sharded) ──────────────
echo ""
echo "  Processing ${#PENDING_DATASETS[@]} dataset(s), each across $N_SHARDS shards..."

TOTAL_OK=0
TOTAL_FAIL=0

for DS in "${PENDING_DATASETS[@]}"; do
    echo ""
    echo "  ── $DS ──────────────────────────────────────────────"

    # Launch N_SHARDS shard processes (one per GPU)
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

    # Wait for all shards
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

    # Merge shards & upload
    echo "    All shards done — merging & uploading..."
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