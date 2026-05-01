#!/bin/bash
# =============================================================================
# launch_convnext_trades.sh
# Top-level launcher: trains convnext_base.fb_in22k with TRADES_v2
# on all 6 datasets, loading HPO configs from full_fine_tuning50.
# Results saved under project: convnext_base_fb_in22k_TRADES_v2
#
# Usage: bash launch_convnext_trades.sh
# =============================================================================

BACKBONE="convnext_base.fb_in22k"
LOSS="TRADES_v2"
SEED=1
PRNM="convnext_base_fb_in22k_TRADES_v2"
HPO_SOURCE_PRNM="full_fine_tuning50"
EMAIL="maxime.heuillet.1@ulaval.ca"          # <-- set your email here
ACCOUNT="aip-adurand"

DATASETS=(
  "flowers-102"
  "stanford_cars"
  "oxford-iiit-pet"
  "caltech101"
  "fgvc-aircraft-2013b"
  "uc-merced-land-use-dataset"
)

mkdir -p ./logs

echo "Submitting 6 training jobs for backbone=${BACKBONE}, loss=${LOSS}"
echo "HPO configs will be loaded from project: ${HPO_SOURCE_PRNM}"
echo "Results will be saved under project: ${PRNM}"
echo ""

for DATA in "${DATASETS[@]}"; do
  echo "  Submitting: dataset=${DATA}"
  sbatch \
    --account="${ACCOUNT}" \
    --mail-user="${EMAIL}" \
    --export="ALL,ACCOUNT=${ACCOUNT},BCKBN=${BACKBONE},DATA=${DATA},SEED=${SEED},LOSS=${LOSS},PRNM=${PRNM},HPO_SOURCE_PRNM=${HPO_SOURCE_PRNM},EMAIL=${EMAIL}" \
    ./job2_train_convnext.sh
done

echo ""
echo "All 6 training jobs submitted."
echo "Each will automatically chain a test job (job3_test_robustgenbench.sh) on success."