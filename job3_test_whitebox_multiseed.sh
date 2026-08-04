#!/bin/bash
# =============================================================================
# job3_test_whitebox_multiseed.sh
# SLURM test job: white-box AutoAttack evaluation (Linf, L2, L1, common) run
# directly against the trained model in a single job/single model load,
# instead of chaining job3_test_linf -> job4_test_l1 -> job5_test_l2 ->
# job6_test_common as 4 separate jobs. Uses the H100-tuned batch sizes from
# Setup.whitebox_eval_batch_size().
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
echo "  JOB: Test (white-box AutoAttack, combined Linf/L2/L1/common)"
echo "  Backbone : ${BCKBN}"
echo "  Dataset  : ${DATA}"
echo "  Loss     : ${LOSS}"
echo "  Project  : ${PRNM}"
echo "  Seed     : ${SEED}"
echo "  Node     : $(hostname)"
echo "  GPUs     : ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# --- Download clean test data from HuggingFace (mirrors job2_train_multiseed.sh) ---
echo "Downloading test data for ${DATA}..."
python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

dataset   = os.environ["DATA"]
hf_repo   = "legolasflagstaff/RobustGenBench"
data_dir  = os.path.join(os.environ.get("SLURM_TMPDIR", "/tmp"), "datasets")
archive   = f"{dataset}_processed.tar.zst"
dest_path = os.path.join(data_dir, archive)

if os.path.exists(dest_path):
    print(f"  Archive already present: {dest_path}")
    sys.exit(0)

os.makedirs(data_dir, exist_ok=True)
print(f"  Downloading {archive} from {hf_repo} ...")
hf_hub_download(
    repo_id=hf_repo,
    repo_type="dataset",
    filename=archive,
    local_dir=data_dir,
)
print(f"  Done. Archive at {dest_path}")
PYEOF

dl_exit=$?
if [ $dl_exit -ne 0 ]; then
    echo "ERROR: Test data download failed (exit ${dl_exit}). Aborting."
    exit 1
fi

# --- Download backbone weights if not already present (mirrors job2_train_multiseed.sh) ---
echo "Checking backbone weights..."
python - <<'PYEOF'
import os, timm, torch
from pathlib import Path

backbone = os.environ["BCKBN"]
dest_dir = Path(os.path.expanduser("~/links/scratch/mheuill/my_backbones/"))
dest_dir.mkdir(parents=True, exist_ok=True)
dest_path = dest_dir / f"{backbone}.pt"

if dest_path.exists():
    print(f"  Backbone weights already exist: {dest_path}")
else:
    print(f"  Downloading {backbone} from timm...")
    model = timm.create_model(backbone, pretrained=True)
    torch.save(model.state_dict(), dest_path)
    print(f"  Saved to {dest_path}")
PYEOF

echo "Starting white-box evaluation..."
python ./distributed_experiment_final.py \
    --mode "test-all" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    --hpo_source_project "${HPO_SOURCE_PRNM}" \
    > stdout_whitebox_"$SLURM_JOB_ID" 2> stderr_whitebox_"$SLURM_JOB_ID"

exit_code=$?
echo "Test exit code: ${exit_code}"

if [ ${exit_code} -ne 0 ]; then
    echo "Test failed. Check stderr_whitebox_${SLURM_JOB_ID} for details."
    exit 1
fi

echo "All white-box evaluations complete for dataset=${DATA}, project=${PRNM}."
