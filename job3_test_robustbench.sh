#!/bin/bash
# =============================================================================
# job3_test_robustgenbench.sh
# SLURM test job: evaluates trained model on RobustGenBench adversarial sets.
# - Downloads .tar.zst archives from HuggingFace at runtime
# - Evaluates across 4 surrogates × 3 threat models + common = 13 test configs
# - Uses 4 GPUs, 3h wall time
#
# Required env vars (passed via --export):
#   ACCOUNT, BCKBN, DATA, SEED, LOSS, PRNM, EMAIL
# =============================================================================

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:58:00
#SBATCH --mail-type=ALL

source ./execute_setup.sh
source ./setup_paths.sh

echo "============================================"
echo "  JOB: Test (RobustGenBench)"
echo "  Backbone : ${BCKBN}"
echo "  Dataset  : ${DATA}"
echo "  Loss     : ${LOSS}"
echo "  Project  : ${PRNM}"
echo "  Node     : $(hostname)"
echo "  GPUs     : ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# --- Download + evaluate ---
python ./evaluate_robustgenbench.py \
    --backbone "${BCKBN}" \
    --dataset  "${DATA}" \
    --loss     "${LOSS}" \
    --seed     "${SEED}" \
    --project  "${PRNM}" \
    > stdout_test_"${SLURM_JOB_ID}" 2> stderr_test_"${SLURM_JOB_ID}"

exit_code=$?
echo "Test exit code: ${exit_code}"

if [ ${exit_code} -ne 0 ]; then
    echo "Test failed. Check stderr_test_${SLURM_JOB_ID} for details."
    exit 1
fi

echo "All evaluations complete for dataset=${DATA}, project=${PRNM}."