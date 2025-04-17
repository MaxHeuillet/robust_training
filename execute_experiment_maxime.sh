#!/bin/bash

# Define variables
seed=1

datas=( 
        'stanford_cars'
        # 'oxford-iiit-pet'
        # 'caltech101' 
        # 'flowers-102' 
        # 'fgvc-aircraft-2013b'
        # 'uc-merced-land-use-dataset' 
        ) 
 
losses=( 
        'CLASSIC_AT' 
        'TRADES_v2'  
         )  

# Two backbone groups
# Two backbone groups
block_1=(
  'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K'
  'CLIP-convnext_base_w-laion2B-s13B-b82K'
  'deit_small_patch16_224.fb_in1k'
  'robust_resnet50'
  'vit_small_patch16_224.augreg_in21k'
  'convnext_base.fb_in1k'
  'resnet50.a1_in1k'
  'robust_vit_base_patch16_224'
  'vit_base_patch16_224.mae'
  'vit_small_patch16_224.dino'
  'convnext_base.fb_in22k'
) 

block_2=(
  'robust_convnext_base'
  'vit_base_patch16_224.augreg_in1k'
  'vit_base_patch16_224.augreg_in21k'
  'vit_base_patch16_clip_224.laion2b'
  'convnext_tiny.fb_in1k'
  'robust_convnext_tiny'
  'robust_deit_small_patch16_224'
  'vit_small_patch16_224.augreg_in1k'
  'convnext_tiny.fb_in22k'
  'vit_base_patch16_clip_224.laion2b_ft_in1k'
  'vit_base_patch16_224.augreg_in21k_ft_in1k'
  )

block_3=(
  'vit_small_patch16_224.augreg_in21k_ft_in1k'
  'eva02_base_patch14_224.mim_in22k'
  'eva02_tiny_patch14_224.mim_in22k'
  'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
  'swin_tiny_patch4_window7_224.ms_in1k'
  'convnext_base.clip_laion2b_augreg_ft_in12k_in1k'
  'convnext_base.fb_in22k_ft_in1k'
  'convnext_tiny.fb_in22k_ft_in1k'
  'coatnet_0_rw_224.sw_in1k'
  'coatnet_2_rw_224.sw_in12k_ft_in1k'
  'coatnet_2_rw_224.sw_in12k'
)

block_repair=(
  'convnext_tiny.fb_in1k',
)

#33-4 = 29 + 7 = 36


# Get the project name 
PRNM=$1

if [ -z "$PRNM" ]; then
  echo "Error: Project name (PRNM) is not set. Please provide it as the first argument."
  exit 1
fi

# ---------- Function to submit jobs ----------
submit_jobs() {
  local ACCOUNT="$1"
  shift  # shift out the account name
  local backbones=("$@")  # remaining arguments are backbone list
  for data in "${datas[@]}"; do
    for bckbn in "${backbones[@]}"; do
      for loss in "${losses[@]}"; do
        sbatch --account="$ACCOUNT" --mail-user="$EMAIL" \
          --export=ALL,ACCOUNT="$ACCOUNT",BCKBN="$bckbn",DATA="$data",SEED="$seed",LOSS="$loss",PRNM="$PRNM",EMAIL="$EMAIL" \
          ./job1_hpo.sh
    done
  done
done
}

# ---------- Submit jobs ----------

EMAIL="maxime.heuillet.1@ulaval.ca"

# echo "Submitting block_1 backbone jobs..."
# submit_jobs "rrg-csubakan" "${block_1[@]}"

# echo "Submitting block_2 backbone jobs..."
# submit_jobs "rrg-adurand" "${block_2[@]}"

# echo "Submitting block_3 backbone jobs..."
# submit_jobs "def-adurand" "${block_3[@]}"

echo "Submitting block_repair backbone jobs..."
submit_jobs "rrg-adurand" "${block_repair[@]}"

