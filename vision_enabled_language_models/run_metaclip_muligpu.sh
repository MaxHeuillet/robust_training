#!/usr/bin/env bash
# ============================================================
# run_metaclip_h14_multigpu.sh
# Craft all threat models with MetaCLIP ViT-H/14 (fullcc2.5b)
# using 4 GPUs.
# - Auto-skips datasets already uploaded to HuggingFace
# - Live progress monitoring in terminal
# - Ctrl+C cleanly kills all child processes
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRAFT="$SCRIPT_DIR/craft_adversarial.py"
LOG_DIR="$SCRIPT_DIR/logs/metaclip_h14"
mkdir -p "$LOG_DIR"

export HF_HOME="$HOME/.cache/huggingface"

# ── Graceful Ctrl+C ───────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    echo "  Interrupted — killing all child processes..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null && echo "    Killed PID $PID" || true
    done
    kill -- -$$ 2>/dev/null || true
    echo "  Done."
    exit 1
}
trap cleanup SIGINT SIGTERM

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

ALL_THREAT_MODELS=(linf8 linf30 l2_2 l2_8 l1_75 l1_300)

if [[ $# -gt 0 ]]; then
    ALL_THREAT_MODELS=("$@")
fi

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

# Check if a dataset/threatmodel archive already exists on HuggingFace
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

# Print live progress for all active logs
print_progress() {
    local TM="$1"; shift
    local ACTIVE_DATASETS=("$@")
    echo ""
    echo "  ┌─ Progress: $TM ──────────────────────────────────────"
    for DS in "${ACTIVE_DATASETS[@]}"; do
        local LOG="$LOG_DIR/${TM}__${DS}.log"
        if [[ ! -f "$LOG" ]]; then
            printf "  │  %-35s  starting...\n" "$DS"
            continue
        fi
        local DONE
        DONE=$(grep -o "[0-9]* images | clean=.*" "$LOG" 2>/dev/null | tail -1 || true)
        if [[ -n "$DONE" ]]; then
            printf "  │  %-35s  ✓ %s\n" "$DS" "$DONE"
        else
            local PROG
            PROG=$(grep -o "[0-9]*%|[0-9]*/[0-9]*" "$LOG" 2>/dev/null | tail -1 || true)
            local BATCH
            BATCH=$(grep -oP "\d+/32" "$LOG" 2>/dev/null | tail -1 || true)
            local ETA
            ETA=$(grep -oP "\d+:\d+<\d+:\d+" "$LOG" 2>/dev/null | tail -1 || true)
            printf "  │  %-35s  batch %s  %s\n" "$DS" "${BATCH:-?/32}" "${ETA:+ETA $ETA}"
        fi
    done
    echo "  └──────────────────────────────────────────────────────"
}

wait_for_batch() {
    local TM="$1"; shift
    local BATCH_DATASETS=("$@")
    local FAILED=0

    # Monitor loop while waiting
    while true; do
        local ALL_DONE=1
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                ALL_DONE=0
                break
            fi
        done

        print_progress "$TM" "${BATCH_DATASETS[@]}"

        if [[ $ALL_DONE -eq 1 ]]; then
            break
        fi
        sleep 30
    done

    for PID in "${PIDS[@]}"; do
        wait "$PID" || FAILED=1
    done
    PIDS=()
    return $FAILED
}

# ── Summary ────────────────────────────────────────────────────
echo "============================================================"
echo "  Surrogate    : MetaCLIP ViT-H/14 fullcc2.5b"
echo "  Threat models: ${ALL_THREAT_MODELS[*]}"
echo "  GPUs         : $N_GPUS"
echo "  Datasets     : ${DATASETS[*]}"
echo "  Logs         : $LOG_DIR"
echo "============================================================"

# ── Pre-download (sequential, avoids race condition) ───────────
echo ""
echo "  Step 1/2 — ensuring all datasets are downloaded..."
CUDA_VISIBLE_DEVICES=0 python "$CRAFT" \
    --surrogate clip --norm Linf --eps 8 --max_samples 1 \
    2>&1 | grep -E "Download|already present|Extracting|ERROR" || true
echo "  Data ready."
echo ""

# ── Main loop ──────────────────────────────────────────────────
echo "  Step 2/2 — crafting adversarial examples..."

TOTAL_SKIPPED=0
TOTAL_LAUNCHED=0

for THREAT_MODEL in "${ALL_THREAT_MODELS[@]}"; do
    READ_ARGS=($(parse_threat_model "$THREAT_MODEL"))
    NORM="${READ_ARGS[0]}"
    EPS="${READ_ARGS[1]}"
    TM_SLUG=$(threat_model_slug "$THREAT_MODEL")

    echo ""
    echo "============================================================"
    echo "  Threat model : $NORM eps=$EPS  ($TM_SLUG)"
    echo "============================================================"

    # Check which datasets are already on HF
    echo "  Checking HuggingFace for existing uploads..."
    PENDING_DATASETS=()
    for DS in "${DATASETS[@]}"; do
        if already_on_hf "$DS" "$TM_SLUG"; then
            echo "    ✓ $DS — already on HF, skipping"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        else
            echo "    ○ $DS — needs processing"
            PENDING_DATASETS+=("$DS")
        fi
    done

    if [[ ${#PENDING_DATASETS[@]} -eq 0 ]]; then
        echo "  All datasets already uploaded — skipping threat model."
        continue
    fi

    echo ""
    echo "  Launching ${#PENDING_DATASETS[@]} dataset(s) across $N_GPUS GPUs..."

    PIDS=()
    GPU_IDX=0
    BATCH_DATASETS=()

    for DS in "${PENDING_DATASETS[@]}"; do
        LOG_FILE="$LOG_DIR/${THREAT_MODEL}__${DS}.log"
        echo "    GPU $GPU_IDX ← $DS"

        CUDA_VISIBLE_DEVICES=$GPU_IDX HF_TOKEN="$HF_TOKEN" \
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
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))
        GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))

        if [[ ${#PIDS[@]} -eq $N_GPUS ]]; then
            echo ""
            echo "  All $N_GPUS GPU slots filled — monitoring..."
            wait_for_batch "$THREAT_MODEL" "${BATCH_DATASETS[@]}" \
                || echo "  WARNING: one or more jobs failed — check logs"
            BATCH_DATASETS=()
            GPU_IDX=0
        fi
    done

    # Remaining jobs
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "  Monitoring final batch of ${#PIDS[@]} job(s)..."
        wait_for_batch "$THREAT_MODEL" "${BATCH_DATASETS[@]}" \
            || echo "  WARNING: one or more jobs failed — check logs"
    fi

    echo "  ✓ $NORM eps=$EPS — complete"
done

echo ""
echo "============================================================"
echo "  All done."
echo "  Skipped (already on HF) : $TOTAL_SKIPPED"
echo "  Launched                : $TOTAL_LAUNCHED"
echo "  Logs                    : $LOG_DIR"
echo "============================================================"