#!/bin/bash
# =============================================================================
# launch_top3_tiny_multiseed.sh
# Reviewer-response launcher: re-trains the top-3 FFT-50 "Tiny" leaderboard
# configurations (see latex_tables/FFT_50_epochs.tex) over 3 random seeds each,
# to check whether the leaderboard ranking holds across seeds.
#
#   Gold   : coat_tiny.in1k          + TRADES_v2   (GR:18)
#   Silver : edgenext_small.usi_in1k + TRADES_v2   (GR:23)
#   Bronze : edgenext_small.usi_in1k + CLASSIC_AT  (GR:33)
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
#   coat_tiny_in1k_TRADES_v2_seed1
#   coat_tiny_in1k_TRADES_v2_seed2
#   coat_tiny_in1k_TRADES_v2_seed3
#
# Usage: bash launch_top3_tiny_multiseed.sh
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
  "coat_tiny.in1k|TRADES_v2|coat_tiny_in1k_TRADES_v2"
  "edgenext_small.usi_in1k|TRADES_v2|edgenext_small_usi_in1k_TRADES_v2"
  "edgenext_small.usi_in1k|CLASSIC_AT|edgenext_small_usi_in1k_CLASSIC_AT"
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
