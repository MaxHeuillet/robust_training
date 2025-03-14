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
  'convnext_tiny.fb_in1k' #this is where we will put all the list of backbones
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

