#!/bin/bash
# =============================================================================
# launch_top3_small_multiseed.sh
# Reviewer-response launcher: re-trains the top-3 FFT-50 "Small" leaderboard
# configurations (see latex_tables/FFT_50_epochs.tex) over 3 random seeds each,
# to check whether the leaderboard ranking holds across seeds.
#
#   Gold   : convnext_tiny.fb_in22k_ft_in1k + TRADES_v2  (GR:14)
#   Silver : convnext_tiny.fb_in1k          + TRADES_v2  (GR:15)
#   Bronze : convnext_tiny.fb_in22k         + TRADES_v2  (GR:20)
#
# HPO configs are NOT re-tuned: each run loads the already-optimal
# hyperparameters from HPO_results/full_fine_tuning50/, same as the original
# leaderboard run. Only the training seed varies.
#
# Test jobs (chained automatically) also save per-observation predictions
# (--save_predictions in evaluate_robustgenbench.py) so accuracy can be
# bootstrapped over observations in addition to being compared across seeds.
#
# Results are saved under project: {PRNM_BASE}_seed{SEED}, e.g.
#   convnext_tiny_fb_in22k_ft_in1k_TRADES_v2_seed1
#   convnext_tiny_fb_in22k_ft_in1k_TRADES_v2_seed2
#   convnext_tiny_fb_in22k_ft_in1k_TRADES_v2_seed3
#
# Usage: bash launch_top3_small_multiseed.sh
# =============================================================================

HPO_SOURCE_PRNM="full_fine_tuning50"
EMAIL="maxime.heuillet.1@ulaval.ca"          # <-- set your email here
ACCOUNT="aip-adurand"

SEEDS=(1 2 3)

DATASETS=(
  "flowers-102"
  "stanford_cars"
  "oxford-iiit-pet"
  "caltech101"
  "fgvc-aircraft-2013b"
  "uc-merced-land-use-dataset"
)

# backbone | loss | base project name (medal, GR from FFT_50_epochs.tex)
CONFIGS=(
  "convnext_tiny.fb_in22k_ft_in1k|TRADES_v2|convnext_tiny_fb_in22k_ft_in1k_TRADES_v2"
  "convnext_tiny.fb_in1k|TRADES_v2|convnext_tiny_fb_in1k_TRADES_v2"
  "convnext_tiny.fb_in22k|TRADES_v2|convnext_tiny_fb_in22k_TRADES_v2"
)

mkdir -p ./logs

for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r BACKBONE LOSS PRNM_BASE <<< "${CFG}"

  for SEED in "${SEEDS[@]}"; do
    PRNM="${PRNM_BASE}_seed${SEED}"

    echo "Submitting backbone=${BACKBONE}, loss=${LOSS}, seed=${SEED} -> project=${PRNM}"
    echo "HPO configs will be loaded from project: ${HPO_SOURCE_PRNM}"

    for DATA in "${DATASETS[@]}"; do
      echo "  Submitting: dataset=${DATA}"
      sbatch \
        --account="${ACCOUNT}" \
        --mail-user="${EMAIL}" \
        --export="ALL,ACCOUNT=${ACCOUNT},BCKBN=${BACKBONE},DATA=${DATA},SEED=${SEED},LOSS=${LOSS},PRNM=${PRNM},HPO_SOURCE_PRNM=${HPO_SOURCE_PRNM},EMAIL=${EMAIL}" \
        ./job2_train_multiseed.sh
    done
  done
  echo ""
done

echo "All jobs submitted: ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds x ${#DATASETS[@]} datasets = $(( ${#CONFIGS[@]} * ${#SEEDS[@]} * ${#DATASETS[@]} )) training jobs."
echo "Each will automatically chain a test job (job3_test_robustgenbench_multiseed.sh) on success,"
echo "which also saves per-observation predictions under {results}/{project}/predictions/ for bootstrap analysis."
