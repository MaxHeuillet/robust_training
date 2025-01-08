#!/bin/bash

# Define variables
seeds=1

datas=(  'caltech101' )
      # 'oxford-iiit-pet' 'flowers-102'  'dtd' 'stanford_cars'  'fgvc-aircraft-2013b'     ) #'imagenette2' 'eurosat' 

losses=( 'TRADES_v2' ) #'CLASSIC_AT'  

# backbones=(
#   'convnext_tiny' 'robust_convnext_tiny' 'convnext_tiny.fb_in22k' 
#   'deit_small_patch16_224.fb_in1k' 'robust_deit_small_patch16_224'
#   'convnext_base' 'convnext_base.fb_in22k' 'robust_convnext_base' 
#   'convnext_base.clip_laion2b' 'convnext_base.clip_laion2b_augreg'
#   'vit_base_patch16_224.augreg_in1k' 'vit_base_patch16_224.augreg_in21k',
#   'vit_base_patch16_224.dino' 'vit_base_patch16_224.mae' 'vit_base_patch16_224.orig_in21k'
#   'vit_base_patch16_224.sam_in1k' 'vit_base_patch16_224_miil.in21k'
#     ) 

backbones=(
  'vit_base_patch16_224.sam_in1k' 
   ) 



ft_type=( 'full_fine_tuning' )

tasks=( 'HPO' ) # 'train' 'test'

# Get the project name as the current date in yy-mm-dd-hh format
PRNM=$1

# Check if PRNM is set
if [ -z "$PRNM" ]; then
  echo "Error: Project name (PRNM) is not set. Please provide it as the first argument."
  exit 1
fi

# Loop over architectures, datasets, losses, seeds, backbones, and fine-tuning types
for task in "${tasks[@]}"; do
  for data in "${datas[@]}"; do
    for loss in "${losses[@]}"; do
      for id in $(seq 1 $seeds); do
        for bckbn in "${backbones[@]}"; do
          for fttype in "${ft_type[@]}"; do
              
              if [ "$CC_CLUSTER" = "beluga" ]; then
                  sbatch --export=ALL,\
TASK="$task",\
BCKBN="$bckbn",\
FTTYPE="$fttype",\
DATA="$data",\
SEED="$id",\
LOSS="$loss",\
PRNM="$PRNM" \
./distributed_experiment_beluga.sh
              else
                  sbatch --export=ALL,\
TASK="$task",\
BCKBN="$bckbn",\
FTTYPE="$fttype",\
DATA="$data",\
SEED="$id",\
LOSS="$loss",\
PRNM="$PRNM" \
./distributed_experiment_other.sh
              fi

          done
        done
      done
    done
  done
done
