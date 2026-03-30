#!/bin/bash
# submit_l1_jobs.sh
# Submit one SLURM job per remaining dataset for L1 eps=75 and L1 eps=300.
# Each job uses all 4 GPUs on a single node (one shard per GPU).
#
# Usage:
#   bash submit_l1_jobs.sh
#
# Prerequisites:
#   source ./execute_setup.sh   (run this BEFORE calling this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/slurm_l1"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Remaining work
# ---------------------------------------------------------------------------
# L1 eps=75  — done: fgvc-aircraft-2013b, flowers-102, oxford-iiit-pet
#              remaining: caltech101 (still running — will be skipped by shard_done flag),
#                         stanford_cars, uc-merced-land-use-dataset
# L1 eps=300 — remaining: all 6 datasets

L1_75_DATASETS=(
    "caltech101"
    "stanford_cars"
    "uc-merced-land-use-dataset"
)

L1_300_DATASETS=(
    "caltech101"
    "fgvc-aircraft-2013b"
    "flowers-102"
    "oxford-iiit-pet"
    "stanford_cars"
    "uc-merced-land-use-dataset"
)

N_SHARDS=4
BATCH_SIZE=128
SURROGATE="clip_vith14"
TIME_LIMIT="02:59:00"

# ---------------------------------------------------------------------------
# Helper: submit one job for one dataset + threat model
# ---------------------------------------------------------------------------
submit_job() {
    local DATASET="$1"
    local NORM="$2"
    local EPS="$3"
    local SLUG="${NORM,,}_eps${EPS}"   # e.g. l1_eps75

    local JOB_NAME="craft_${SLUG}__${DATASET}"
    local OUT_LOG="$LOG_DIR/slurm-%j__${SLUG}__${DATASET}.out"
    local ERR_LOG="$LOG_DIR/slurm-%j__${SLUG}__${DATASET}.err"

    echo "  Submitting: $DATASET  $NORM eps=$EPS"

    sbatch \
        --account=aip-adurand \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=48 \
        --gres=gpu:h100:4 \
        --mem=480G \
        --time="$TIME_LIMIT" \
        --job-name="$JOB_NAME" \
        --mail-type=ALL \
        --output="$OUT_LOG" \
        --error="$ERR_LOG" \
        --wrap="
set -euo pipefail
source $SCRIPT_DIR/execute_setup.sh

echo \"[\$(date)] Starting $DATASET $NORM eps=$EPS on \$(hostname)\"
echo \"GPUs available: \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')\"

# Step 1 — ensure data is downloaded (only rank 0 does this, others wait)
python $SCRIPT_DIR/craft_shard.py \
    --dataset   $DATASET \
    --norm      $NORM \
    --eps       $EPS \
    --surrogate $SURROGATE \
    --shard_idx 0 \
    --n_shards  1 \
    --gpu       0 \
    --batch_size 1 \
    2>&1 | grep -E 'Download|already present|Extracting|ERROR' || true

# Step 2 — run 4 shards in parallel (one per GPU)
echo \"[\$(date)] Launching 4 shards...\"
for SHARD_IDX in 0 1 2 3; do
    python $SCRIPT_DIR/craft_shard.py \
        --dataset    $DATASET \
        --norm       $NORM \
        --eps        $EPS \
        --surrogate  $SURROGATE \
        --batch_size $BATCH_SIZE \
        --shard_idx  \$SHARD_IDX \
        --n_shards   $N_SHARDS \
        --gpu        \$SHARD_IDX \
        > $LOG_DIR/${SLUG}__${DATASET}__shard\${SHARD_IDX}.log 2>&1 &
done

echo \"[\$(date)] Waiting for all shards to complete...\"
wait
echo \"[\$(date)] All shards done — merging and uploading\"

# Step 3 — merge shards and upload to HuggingFace
python $SCRIPT_DIR/craft_shard.py \
    --dataset    $DATASET \
    --norm       $NORM \
    --eps        $EPS \
    --surrogate  $SURROGATE \
    --merge \
    --upload_hf

echo \"[\$(date)] Done: $DATASET $NORM eps=$EPS\"
"
    echo "    → submitted"
}

# ---------------------------------------------------------------------------
# Submit all remaining jobs
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Submitting L1 adversarial crafting jobs"
echo "  Node config : 1 node, 4x H100, 48 CPUs, 480G RAM"
echo "  Time limit  : $TIME_LIMIT"
echo "  Shards      : $N_SHARDS (one per GPU)"
echo "  Batch size  : $BATCH_SIZE per GPU"
echo "============================================================"

echo ""
echo "── L1 eps=75 (remaining datasets) ─────────────────────────"
for DS in "${L1_75_DATASETS[@]}"; do
    submit_job "$DS" "L1" "75"
done

echo ""
echo "── L1 eps=300 (all datasets) ───────────────────────────────"
for DS in "${L1_300_DATASETS[@]}"; do
    submit_job "$DS" "L1" "300"
done

echo ""
echo "============================================================"
echo "  All jobs submitted."
echo "  Monitor with:  squeue -u \$USER"
echo "  Logs at:       $LOG_DIR/"
echo "============================================================"
echo ""
echo "  After jobs complete, check shard logs with:"
echo "    ls $LOG_DIR/*.log | xargs -I{} sh -c 'echo \"=== {} ===\"; tail -3 {}'"