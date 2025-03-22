#!/bin/bash

# Define variables
seeds=1

datas=( 
        'stanford_cars'
        # 'oxford-iiit-pet'
        # 'caltech101' 
        # 'flowers-102' 
        # 'fgvc-aircraft-2013b'
        # 'uc-merced-land-use-dataset' 
        ) 
 
losses=( 'CLASSIC_AT' 'TRADES_v2'   )  

# Two backbone groups
scientific_backbones=(
  # 'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K'
  # 'CLIP-convnext_base_w-laion2B-s13B-b82K'
  # 'deit_small_patch16_224.fb_in1k'
  # 'robust_resnet50'
  # 'vit_base_patch16_224.mae'
  # 'vit_small_patch16_224.augreg_in21k'
  # 'convnext_base.fb_in1k'
  'resnet50.a1_in1k'
  # 'robust_vit_base_patch16_224'
  # 'vit_base_patch16_224.sam_in1k'
  # 'vit_small_patch16_224.dino'
  # 'convnext_base.fb_in22k'
  # 'robust_convnext_base'
  # 'vit_base_patch16_224.augreg_in1k'
  # 'vit_base_patch16_224.augreg_in21k'
  # 'vit_base_patch16_224.dino'
  # 'vit_base_patch16_clip_224.laion2b'
  'convnext_tiny.fb_in1k'
  # 'robust_convnext_tiny'
  # 'robust_deit_small_patch16_224'
  # 'vit_base_patch16_224.dino'
  'vit_small_patch16_224.augreg_in1k'
  # 'convnext_tiny.fb_in22k'
)

performance_backbones=(
  # 'vit_base_patch16_clip_224.laion2b_ft_in1k'
  # 'vit_base_patch16_224.augreg_in21k_ft_in1k'
  # 'vit_small_patch16_224.augreg_in21k_ft_in1k'
  # 'eva02_base_patch14_224.mim_in22k'
  'eva02_tiny_patch14_224.mim_in22k'
  'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
  # 'swin_tiny_patch4_window7_224.ms_in1k'
  # 'convnext_base.clip_laion2b_augreg_ft_in12k_in1k'
  # 'convnext_base.fb_in22k_ft_in1k'
  # 'convnext_tiny.fb_in22k_ft_in1k'
)

# Get the project name 
PRNM=$1

if [ -z "$PRNM" ]; then
  echo "Error: Project name (PRNM) is not set. Please provide it as the first argument."
  exit 1
fi

# ---------- Function to submit jobs ----------
submit_jobs() {
  local backbones=("$@")  # All arguments passed to this function (backbone list)
  for id in $(seq 1 $seeds); do
    for data in "${datas[@]}"; do
      for bckbn in "${backbones[@]}"; do
        for loss in "${losses[@]}"; do
          sbatch --export=ALL,\
BCKBN="$bckbn",\
DATA="$data",\
SEED="$id",\
LOSS="$loss",\
PRNM="$PRNM" \
./job1_hpo.sh
        done
      done
    done
  done
}

# ---------- Submit jobs ----------

echo "Submitting SCIENTIFIC backbone jobs..."
submit_jobs "${scientific_backbones[@]}"

echo "Submitting PERFORMANCE backbone jobs..."
submit_jobs "${performance_backbones[@]}"