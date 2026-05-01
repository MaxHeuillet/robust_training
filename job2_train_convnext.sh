#!/bin/bash
# =============================================================================
# job2_train_convnext.sh
# SLURM training job for convnext_base.fb_in22k with TRADES_v2.
# - Skips HPO: loads optimal config from HPO_results/{HPO_SOURCE_PRNM}/
# - Downloads training data from HuggingFace at runtime
# - Chains job3_test_robustgenbench.sh on success
#
# Required env vars (passed via --export):
#   ACCOUNT, BCKBN, DATA, SEED, LOSS, PRNM, HPO_SOURCE_PRNM, EMAIL
# =============================================================================

#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=02:59:00
#SBATCH --mail-type=ALL
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

source ./execute_setup.sh
source ./setup_paths.sh

echo "============================================"
echo "  JOB: Training (skip HPO)"
echo "  Backbone : ${BCKBN}"
echo "  Dataset  : ${DATA}"
echo "  Loss     : ${LOSS}"
echo "  Project  : ${PRNM}"
echo "  HPO src  : ${HPO_SOURCE_PRNM}"
echo "  Seed     : ${SEED}"
echo "  Node     : $(hostname)"
echo "  GPUs     : ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# --- Download training data from HuggingFace ---
echo "Downloading training data for ${DATA}..."
python - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

dataset  = os.environ["DATA"]
hf_repo  = "legolasflagstaff/RobustGenBench"
data_dir = os.path.expandvars(os.environ.get("DATASET_PATH", "/tmp/robustgenbench/data_processed"))

sentinel = os.path.join(data_dir, f".{dataset}_downloaded")
if os.path.exists(sentinel):
    print(f"  Training data for {dataset} already present, skipping download.")
    sys.exit(0)

os.makedirs(data_dir, exist_ok=True)
print(f"  Downloading {dataset} from {hf_repo} ...")
snapshot_download(
    repo_id=hf_repo,
    repo_type="dataset",
    local_dir=data_dir,
    allow_patterns=[f"{dataset}_processed.tar.zst", "class_names/*"],
    ignore_patterns=["adversarial/*"],
)
open(sentinel, "w").close()
print(f"  Done. Data at {data_dir}")
PYEOF

dl_exit=$?
if [ $dl_exit -ne 0 ]; then
    echo "ERROR: Training data download failed (exit ${dl_exit}). Aborting."
    exit 1
fi

# --- Training step (mode=train, loads HPO yaml from HPO_SOURCE_PRNM) ---
echo "Starting training..."
python ./distributed_experiment_final.py \
    --mode "train" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    --hpo_source_project "${HPO_SOURCE_PRNM}" \
    > stdout_train_"${SLURM_JOB_ID}" 2> stderr_train_"${SLURM_JOB_ID}"

exit_code=$?
echo "Training exit code: ${exit_code}"

# --- Chain test job on success ---
if [ ${exit_code} -eq 0 ]; then
    echo "Training succeeded. Submitting test job..."
    sbatch \
      --account="${ACCOUNT}" \
      --mail-user="${EMAIL}" \
      --export="ALL,ACCOUNT=${ACCOUNT},BCKBN=${BCKBN},DATA=${DATA},SEED=${SEED},LOSS=${LOSS},PRNM=${PRNM},EMAIL=${EMAIL}" \
      ./job3_test_robustgenbench.sh
else
    echo "Training failed. No test job submitted."
    echo "Check stderr_train_${SLURM_JOB_ID} for details."
    exit 1
fi