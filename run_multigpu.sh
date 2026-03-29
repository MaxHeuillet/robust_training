#!/usr/bin/env bash
# ============================================================
# run_clip_vith14_multigpu.sh
# Craft all threat models with CLIP ViT-H/14 using 4 GPUs.
# Ctrl+C cleanly kills all child processes.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRAFT="$SCRIPT_DIR/craft_adversarial.py"
LOG_DIR="$SCRIPT_DIR/logs/clip_vith14"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"

# ── Graceful Ctrl+C — kill all child processes ────────────────
PIDS=()
cleanup() {
    echo ""
    echo "Interrupted — killing all child processes..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null && echo "  Killed PID $PID" || true
    done
    # Also kill any lingering python processes in this process group
    kill -- -$$ 2>/dev/null || true
    echo "Done."
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── HF token ──────────────────────────────────────────────────
TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -f "$TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$TOKEN_FILE")"
    echo "  HF token : found"
else
    echo "  WARNING: no HF token at $TOKEN_FILE — downloads may be rate-limited"
    echo "  Run: huggingface-cli login"
fi

SURROGATE="clip_vith14"
BATCH_SIZE=32
N_GPUS=4

DATASETS=(
    "caltech101"
    "fgvc-aircraft-2013b"
    "flowers-102"
    "oxford-iiit-pet"
    "stanford_cars"
    "uc-merced-land-use-dataset"
)

ALL_THREAT_MODELS=(linf8 linf30 l2_2 l2_8 l1_75 l1_300)

if [[ $# -gt 0 ]]; then
    ALL_THREAT_MODELS=("$@")
fi

echo "============================================================"
echo "  Surrogate    : CLIP ViT-H/14 LAION-2B"
echo "  Threat models: ${ALL_THREAT_MODELS[*]}"
echo "  GPUs         : $N_GPUS"
echo "  Datasets     : ${DATASETS[*]}"
echo "  Logs         : $LOG_DIR"
echo "============================================================"

# ── Pre-download ALL data sequentially ───────────────────────
echo ""
echo "Step 1/2 — pre-downloading datasets (sequential)..."
CUDA_VISIBLE_DEVICES=0 python "$CRAFT" \
    --surrogate clip \
    --norm Linf --eps 8 \
    --max_samples 1 \
    2>&1 | grep -E "Download|already present|Extracting|ERROR" || true
echo "  Data ready."
echo ""

parse_threat_model() {
    case "$1" in
        linf8)   echo "Linf 8"   ;;
        linf30)  echo "Linf 30"  ;;
        l2_2)    echo "L2 2.0"   ;;
        l2_8)    echo "L2 8.0"   ;;
        l1_75)   echo "L1 75"    ;;
        l1_300)  echo "L1 300"   ;;
        *) echo "ERROR: unknown threat model $1" >&2; exit 1 ;;
    esac
}

wait_for_batch() {
    local FAILED=0
    for PID in "${PIDS[@]}"; do
        if wait "$PID"; then
            echo "    PID $PID OK"
        else
            echo "    PID $PID FAILED — check $LOG_DIR"
            FAILED=1
        fi
    done
    PIDS=()
    return $FAILED
}

echo "Step 2/2 — crafting adversarial examples (parallel)..."

for THREAT_MODEL in "${ALL_THREAT_MODELS[@]}"; do
    READ_ARGS=($(parse_threat_model "$THREAT_MODEL"))
    NORM="${READ_ARGS[0]}"
    EPS="${READ_ARGS[1]}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Threat model : $NORM eps=$EPS"
    echo "------------------------------------------------------------"

    PIDS=()
    GPU_IDX=0

    for DATASET in "${DATASETS[@]}"; do
        LOG_FILE="$LOG_DIR/${THREAT_MODEL}__${DATASET}.log"
        echo "  GPU $GPU_IDX ← $DATASET"

        CUDA_VISIBLE_DEVICES=$GPU_IDX \
        HF_TOKEN="$HF_TOKEN" \
        python "$CRAFT" \
            --surrogate  "$SURROGATE" \
            --norm       "$NORM" \
            --eps        "$EPS" \
            --batch_size "$BATCH_SIZE" \
            --dataset    "$DATASET" \
            --upload_hf \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))

        if [[ ${#PIDS[@]} -eq $N_GPUS ]]; then
            echo "  Waiting for batch of $N_GPUS..."
            wait_for_batch || echo "  WARNING: one or more jobs failed, continuing..."
            GPU_IDX=0
        fi
    done

    # Wait for remaining jobs
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "  Waiting for remaining ${#PIDS[@]} job(s)..."
        wait_for_batch || echo "  WARNING: one or more jobs failed."
    fi

    echo "  ✓ $NORM eps=$EPS complete"
done

echo ""
echo "All threat models complete."
echo "Logs: $LOG_DIR"