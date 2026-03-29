#!/usr/bin/env bash
# ============================================================
# run_clip_vith14_multigpu.sh
# Craft all threat models with CLIP ViT-H/14 using 4 GPUs.
#
# Strategy: dispatch each dataset to a GPU slot (0-3).
# Up to 4 datasets run in parallel per threat model.
# Waits for all 4 to finish before starting the next batch.
#
# Usage:
#   bash run_clip_vith14_multigpu.sh              # all threat models
#   bash run_clip_vith14_multigpu.sh linf8 linf30 # specific models
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRAFT="$SCRIPT_DIR/craft_adversarial.py"
LOG_DIR="$SCRIPT_DIR/logs/clip_vith14"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"

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

# If arguments provided, run only those threat models
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

for THREAT_MODEL in "${ALL_THREAT_MODELS[@]}"; do
    READ_ARGS=($(parse_threat_model "$THREAT_MODEL"))
    NORM="${READ_ARGS[0]}"
    EPS="${READ_ARGS[1]}"

    echo ""
    echo "============================================================"
    echo "  Threat model : $NORM eps=$EPS"
    echo "============================================================"

    PIDS=()
    GPU_IDX=0

    for DATASET in "${DATASETS[@]}"; do
        LOG_FILE="$LOG_DIR/${THREAT_MODEL}__${DATASET}.log"
        echo "  GPU $GPU_IDX ← $DATASET  (log: $LOG_FILE)"

        CUDA_VISIBLE_DEVICES=$GPU_IDX python "$CRAFT" \
            --surrogate  "$SURROGATE" \
            --norm       "$NORM" \
            --eps        "$EPS" \
            --batch_size "$BATCH_SIZE" \
            --dataset    "$DATASET" \
            --upload_hf \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))

        # If we've filled all GPU slots, wait for the current batch to finish
        if [[ ${#PIDS[@]} -eq $N_GPUS ]]; then
            echo "  Waiting for batch of $N_GPUS to complete..."
            for PID in "${PIDS[@]}"; do
                if wait "$PID"; then
                    echo "    PID $PID finished OK"
                else
                    echo "    PID $PID FAILED — check logs in $LOG_DIR"
                fi
            done
            PIDS=()
            GPU_IDX=0
        fi
    done

    # Wait for any remaining jobs (last batch may be < N_GPUS)
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "  Waiting for remaining ${#PIDS[@]} job(s)..."
        for PID in "${PIDS[@]}"; do
            if wait "$PID"; then
                echo "    PID $PID finished OK"
            else
                echo "    PID $PID FAILED — check logs in $LOG_DIR"
            fi
        done
    fi

    echo "  ✓ $NORM eps=$EPS — all datasets done"
done

echo ""
echo "All threat models complete."
echo "Logs saved to: $LOG_DIR"