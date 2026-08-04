#!/bin/bash
# =============================================================================
# retry_base_failed.sh
# Re-submits the 22 Base FFT-50 (backbone, seed, dataset) combinations that
# failed on the first launch_top3_base_multiseed.sh run due to a stale
# hardcoded path (/home/rishika/...) baked into several HPO_results yaml
# files (now fixed across all 528 affected configs in configs/HPO_results/).
#
# All 22 failures were in the coatnet_2_rw_224 configs (Silver + Bronze);
# the convnext_base.fb_in22k (Gold) config had 0 failures and is not retried.
#
# Usage: bash retry_base_failed.sh
# =============================================================================

HPO_SOURCE_PRNM="full_fine_tuning50"
EMAIL="maxime.heuillet.1@ulaval.ca"
ACCOUNT="aip-adurand"
LOSS="TRADES_v2"

mkdir -p ./logs

submit() {
  local BACKBONE="$1" SEED="$2" DATA="$3" PRNM_BASE="$4"
  local PRNM="${PRNM_BASE}_seed${SEED}"
  echo "Retrying backbone=${BACKBONE}, seed=${SEED}, dataset=${DATA} -> project=${PRNM}"
  sbatch \
    --account="${ACCOUNT}" \
    --mail-user="${EMAIL}" \
    --export="ALL,ACCOUNT=${ACCOUNT},BCKBN=${BACKBONE},DATA=${DATA},SEED=${SEED},LOSS=${LOSS},PRNM=${PRNM},HPO_SOURCE_PRNM=${HPO_SOURCE_PRNM},EMAIL=${EMAIL}" \
    ./job2_train_multiseed.sh
}

SILVER="coatnet_2_rw_224.sw_in12k_ft_in1k"
SILVER_PRNM="coatnet_2_rw_224_sw_in12k_ft_in1k_TRADES_v2"
BRONZE="coatnet_2_rw_224.sw_in12k"
BRONZE_PRNM="coatnet_2_rw_224_sw_in12k_TRADES_v2"

# --- Silver (coatnet_2_rw_224.sw_in12k_ft_in1k): 10 failures ---
submit "$SILVER" 1 "fgvc-aircraft-2013b"          "$SILVER_PRNM"
submit "$SILVER" 1 "uc-merced-land-use-dataset"   "$SILVER_PRNM"
submit "$SILVER" 2 "oxford-iiit-pet"              "$SILVER_PRNM"
submit "$SILVER" 2 "caltech101"                   "$SILVER_PRNM"
submit "$SILVER" 2 "fgvc-aircraft-2013b"          "$SILVER_PRNM"
submit "$SILVER" 2 "uc-merced-land-use-dataset"   "$SILVER_PRNM"
submit "$SILVER" 3 "flowers-102"                  "$SILVER_PRNM"
submit "$SILVER" 3 "caltech101"                   "$SILVER_PRNM"
submit "$SILVER" 3 "fgvc-aircraft-2013b"          "$SILVER_PRNM"
submit "$SILVER" 3 "uc-merced-land-use-dataset"   "$SILVER_PRNM"

# --- Bronze (coatnet_2_rw_224.sw_in12k): 12 failures ---
submit "$BRONZE" 1 "flowers-102"                  "$BRONZE_PRNM"
submit "$BRONZE" 1 "stanford_cars"                "$BRONZE_PRNM"
submit "$BRONZE" 1 "caltech101"                   "$BRONZE_PRNM"
submit "$BRONZE" 1 "fgvc-aircraft-2013b"          "$BRONZE_PRNM"
submit "$BRONZE" 1 "uc-merced-land-use-dataset"   "$BRONZE_PRNM"
submit "$BRONZE" 2 "flowers-102"                  "$BRONZE_PRNM"
submit "$BRONZE" 2 "stanford_cars"                "$BRONZE_PRNM"
submit "$BRONZE" 2 "caltech101"                   "$BRONZE_PRNM"
submit "$BRONZE" 2 "uc-merced-land-use-dataset"   "$BRONZE_PRNM"
submit "$BRONZE" 3 "stanford_cars"                "$BRONZE_PRNM"
submit "$BRONZE" 3 "caltech101"                   "$BRONZE_PRNM"
submit "$BRONZE" 3 "uc-merced-land-use-dataset"   "$BRONZE_PRNM"

echo ""
echo "All 22 retry jobs submitted."
