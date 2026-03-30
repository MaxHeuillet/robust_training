#!/bin/bash
# submit_l1_remaining.sh
# Submit only the 2 remaining datasets for L1 eps=75:
#   - caltech101 (shard3 failed with FileExistsError)
#   - stanford_cars (shard0 failed with FileExistsError)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/slurm_l1"
mkdir -p "$LOG_DIR"

DATASETS=("caltech101" "stanford_cars")
NORM="L1"
EPS="75"
N_SHARDS=4
BATCH_SIZE=64
SURROGATE="clip_vith14"
TIME_LIMIT="02:59:00"
SLUG="l1_eps75"

# Read HF token on login node
HF_TOKEN_VALUE="$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')"
if [[ -z "$HF_TOKEN_VALUE" ]]; then
    echo "  ERROR: HF token not found at ~/.cache/huggingface/token"
    exit 1
fi
echo "  HF token found."

submit_job() {
    local DATASET="$1"
    local JOB_NAME="craft_${SLUG}__${DATASET}__retry"
    local OUT_LOG="$LOG_DIR/slurm-%j__${SLUG}__${DATASET}__retry.out"
    local ERR_LOG="$LOG_DIR/slurm-%j__${SLUG}__${DATASET}__retry.err"
    local JOB_SCRIPT="$LOG_DIR/job_${SLUG}__${DATASET}__retry.sh"

    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=ALL
#SBATCH --output=${OUT_LOG}
#SBATCH --error=${ERR_LOG}

set -euo pipefail
source ${SCRIPT_DIR}/execute_setup.sh

export HF_TOKEN="${HF_TOKEN_VALUE}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN_VALUE}"

echo "[\$(date)] Starting ${DATASET} ${NORM} eps=${EPS} on \$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Clean up any stale shard output from previous failed runs
# so shard_done.json flags don't incorrectly skip re-runs
echo "[\$(date)] Cleaning stale shard dirs..."
rm -rf /tmp/robustgenbench/adversarial_examples/${DATASET}__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard__shard*
rm -rf /tmp/robustgenbench/work/${DATASET}

# Run 4 shards in parallel — lock serializes download + extraction
echo "[\$(date)] Launching 4 shards..."
for SHARD_IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\${SHARD_IDX} python ${SCRIPT_DIR}/craft_shard.py \
        --dataset    ${DATASET} \
        --norm       ${NORM} \
        --eps        ${EPS} \
        --surrogate  ${SURROGATE} \
        --batch_size ${BATCH_SIZE} \
        --shard_idx  \${SHARD_IDX} \
        --n_shards   ${N_SHARDS} \
        > ${LOG_DIR}/${SLUG}__${DATASET}__shard\${SHARD_IDX}.log 2>&1 &
done

echo "[\$(date)] Waiting for all shards to complete..."
wait

echo "[\$(date)] All shards done — merging and uploading"

python ${SCRIPT_DIR}/craft_shard.py \
    --dataset   ${DATASET} \
    --norm      ${NORM} \
    --eps       ${EPS} \
    --surrogate ${SURROGATE} \
    --merge \
    --upload_hf

echo "[\$(date)] Done: ${DATASET} ${NORM} eps=${EPS}"
JOBEOF

    chmod +x "$JOB_SCRIPT"
    echo "  Submitting: $DATASET $NORM eps=$EPS (retry)"
    sbatch "$JOB_SCRIPT"
    echo "    → submitted"
}

echo ""
echo "============================================================"
echo "  Resubmitting failed L1 eps=75 datasets"
echo "  Datasets : ${DATASETS[*]}"
echo "  Note: stale shard dirs will be cleaned before running"
echo "============================================================"
echo ""

for DS in "${DATASETS[@]}"; do
    submit_job "$DS"
done

echo ""
echo "============================================================"
echo "  Done. Monitor with: squeue -u \$USER"
echo "============================================================"