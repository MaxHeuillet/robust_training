#!/usr/bin/env bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=2:59:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/metaclip_h14/%x_%j.out
#SBATCH --error=logs/metaclip_h14/%x_%j.err

# ============================================================
# slurm_metaclip_h14_job.sh
#
# Processes all 6 datasets for ONE threat model using 4 GPUs.
# Submit one job per threat model:
#
#   sbatch -J linf8  slurm_metaclip_h14_job.sh linf8
#   sbatch -J linf30 slurm_metaclip_h14_job.sh linf30
#   sbatch -J l2_2   slurm_metaclip_h14_job.sh l2_2
#   sbatch -J l2_8   slurm_metaclip_h14_job.sh l2_8
#   sbatch -J l1_75  slurm_metaclip_h14_job.sh l1_75
#   sbatch -J l1_300 slurm_metaclip_h14_job.sh l1_300
#
# Or submit all 6 at once:
#   bash submit_all_metaclip_h14.sh
# ============================================================

# Source environment setup before strict mode (it may have non-fatal errors)
source ./execute_setup.sh || true

set -euo pipefail

THREAT_MODEL="${1:?Usage: sbatch slurm_metaclip_h14_job.sh <threat_model>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRAFT="$SCRIPT_DIR/craft_adversarial.py"
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

DATASETS=(
    "caltech101"
    "fgvc-aircraft-2013b"
    "flowers-102"
    "oxford-iiit-pet"
    "stanford_cars"
    "uc-merced-land-use-dataset"
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
echo "  Job          : $SLURM_JOB_ID"
echo "  Node         : $(hostname)"
echo "  Surrogate    : MetaCLIP ViT-H/14 fullcc2.5b"
echo "  Threat model : $NORM eps=$EPS  ($TM_SLUG)"
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

# ── Process datasets in batches of N_GPUS ─────────────────────
echo ""
echo "  Processing ${#PENDING_DATASETS[@]} dataset(s)..."

IDX=0
while [[ $IDX -lt ${#PENDING_DATASETS[@]} ]]; do
    PIDS=()
    BATCH_DATASETS=()
    GPU=0

    # Launch up to N_GPUS datasets in parallel
    while [[ $GPU -lt $N_GPUS && $IDX -lt ${#PENDING_DATASETS[@]} ]]; do
        DS="${PENDING_DATASETS[$IDX]}"
        LOG_FILE="$LOG_DIR/${THREAT_MODEL}__${DS}__${SLURM_JOB_ID}.log"
        echo "    GPU $GPU ← $DS"

        CUDA_VISIBLE_DEVICES=$GPU \
        python "$CRAFT" \
            --surrogate  "$SURROGATE" \
            --norm       "$NORM" \
            --eps        "$EPS" \
            --batch_size "$BATCH_SIZE" \
            --dataset    "$DS" \
            --upload_hf \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        BATCH_DATASETS+=("$DS")
        GPU=$((GPU + 1))
        IDX=$((IDX + 1))
    done

    echo "    Waiting for batch: ${BATCH_DATASETS[*]}"

    FAILED=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || FAILED=$((FAILED + 1))
    done

    if [[ $FAILED -gt 0 ]]; then
        echo "    WARNING: $FAILED job(s) failed in this batch — check logs"
    else
        echo "    ✓ Batch complete: ${BATCH_DATASETS[*]}"
    fi
done

echo ""
echo "============================================================"
echo "  Job $SLURM_JOB_ID complete: $NORM eps=$EPS"
echo "============================================================"