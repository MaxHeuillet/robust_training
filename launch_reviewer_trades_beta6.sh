#!/bin/bash
# =============================================================================
# launch_reviewer_trades_beta6.sh
# Reviewer-response experiment: re-trains 5 architectures with TRADES_v2 using
# the exact HPO-tuned hyperparameters from HPO_results/full_fine_tuning50/,
# with a single change: beta overridden to 6.0 (originally tuned to 1.0).
# See configs/HPO_results/reviewer_TRADES_beta6/ for the modified yamls
# (generated as an exact copy of full_fine_tuning50 with only beta edited).
#
# Architectures:
#   vit_base_patch16_224.augreg_in1k
#   vit_small_patch16_224.augreg_in1k
#   deit_small_patch16_224.fb_in1k
#   swin_tiny_patch4_window7_224.ms_in1k
#   eva02_base_patch14_224.mim_in22k
#
# Single seed (seed=1, no multiseed sweep), all 6 datasets, all RobustGenBench
# perturbations (chained test job, same as the other launch scripts).
#
# Results are saved under project: {backbone_sanitized}_TRADES_beta6, e.g.
#   trained_state_dicts/vit_base_patch16_224_augreg_in1k_TRADES_beta6/
#   robustgenbench_eval/vit_base_patch16_224_augreg_in1k_TRADES_beta6/
#
# Usage: bash launch_reviewer_trades_beta6.sh
# =============================================================================

HPO_SOURCE_PRNM="reviewer_TRADES_beta6"
EMAIL="maxime.heuillet.1@ulaval.ca"
ACCOUNT="aip-adurand"
LOSS="TRADES_v2"
SEED=1

DATASETS=(
  "flowers-102"
  "stanford_cars"
  "oxford-iiit-pet"
  "caltech101"
  "fgvc-aircraft-2013b"
  "uc-merced-land-use-dataset"
)

# backbone | project name
BACKBONES=(
  "vit_base_patch16_224.augreg_in1k|vit_base_patch16_224_augreg_in1k_TRADES_beta6"
  "vit_small_patch16_224.augreg_in1k|vit_small_patch16_224_augreg_in1k_TRADES_beta6"
  "deit_small_patch16_224.fb_in1k|deit_small_patch16_224_fb_in1k_TRADES_beta6"
  "swin_tiny_patch4_window7_224.ms_in1k|swin_tiny_patch4_window7_224_ms_in1k_TRADES_beta6"
  "eva02_base_patch14_224.mim_in22k|eva02_base_patch14_224_mim_in22k_TRADES_beta6"
)

mkdir -p ./logs

for BB in "${BACKBONES[@]}"; do
  IFS='|' read -r BACKBONE PRNM <<< "${BB}"

  echo "Submitting backbone=${BACKBONE}, beta=6.0 -> project=${PRNM}"
  echo "HPO configs will be loaded from project: ${HPO_SOURCE_PRNM}"

  for DATA in "${DATASETS[@]}"; do
    echo "  Submitting: dataset=${DATA}"
    sbatch \
      --account="${ACCOUNT}" \
      --mail-user="${EMAIL}" \
      --export="ALL,ACCOUNT=${ACCOUNT},BCKBN=${BACKBONE},DATA=${DATA},SEED=${SEED},LOSS=${LOSS},PRNM=${PRNM},HPO_SOURCE_PRNM=${HPO_SOURCE_PRNM},EMAIL=${EMAIL}" \
      ./job2_train_multiseed.sh
  done
  echo ""
done

echo "All jobs submitted: ${#BACKBONES[@]} architectures x ${#DATASETS[@]} datasets = $(( ${#BACKBONES[@]} * ${#DATASETS[@]} )) training jobs."
echo "Each will automatically chain a test job (job3_test_robustgenbench_multiseed.sh) on success,"
echo "which also saves per-observation predictions under {results}/{project}/predictions/ for bootstrap analysis."
