#!/bin/bash

# Define variables
seeds=1

datas=( 'stanford_cars'
        'oxford-iiit-pet'
        'caltech101' 
        'flowers-102' 
        'fgvc-aircraft-2013b'
        'uc-merced-land-use-dataset' 
        ) # 'dtd' 
 
losses=( 'CLASSIC_AT' 'TRADES_v2'   )  #  

backbones=(
 'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
 'CLIP-convnext_base_w-laion2B-s13B-b82K',
 'deit_small_patch16_224.fb_in1k', 
 'robust_resnet50', 
 'vit_base_patch16_224.mae', 
 'vit_small_patch16_224.augreg_in21k', 
 'convnext_base.fb_in1k', 
 'resnet50.a1_in1k', 
 'robust_vit_base_patch16_224', 
 'vit_base_patch16_224.sam_in1k', 
 'vit_small_patch16_224.dino', 
 'convnext_base.fb_in22k', 
 'robust_convnext_base', 
 'vit_base_patch16_224.augreg_in1k',
 'vit_base_patch16_224.augreg_in21k', 
 'vit_base_patch16_224.dino', 
 'vit_base_patch16_clip_224.laion2b',
 'convnext_tiny.fb_in1k',
 'robust_convnext_tiny', 
 'robust_deit_small_patch16_224',
 'vit_base_patch16_224.dino', 
 'vit_small_patch16_224.augreg_in1k', 
 'convnext_tiny.fb_in22k' 
     )   

# Get the project name 
PRNM=$1 #you need to set a project name

# Check if PRNM is set
if [ -z "$PRNM" ]; then
  echo "Error: Project name (PRNM) is not set. Please provide it as the first argument."
  exit 1
fi

# Loop over architectures, datasets, losses, seeds, backbones, and fine-tuning types

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
  ./distributed_experiment_final.sh

      done
    done
  done
done

