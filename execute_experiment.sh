#!/bin/bash

# Define variables
seeds=1

datas=( 'uc-merced-land-use-dataset' ) 
        # 'caltech101' 'stanford_cars' 'flowers-102' 
        #  'oxford-iiit-pet'  'dtd'  'fgvc-aircraft-2013b' 'oxford-iiit-pet' 

losses=( 'CLASSIC_AT' 'TRADES_v2'   )  #  

backbones=(
  
   )  



# Get the project name 
PRNM=$1

# Check if PRNM is set
if [ -z "$PRNM" ]; then
  echo "Error: Project name (PRNM) is not set. Please provide it as the first argument."
  exit 1
fi

# Loop over architectures, datasets, losses, seeds, backbones, and fine-tuning types

for data in "${datas[@]}"; do
  for loss in "${losses[@]}"; do
    for id in $(seq 1 $seeds); do
      for bckbn in "${backbones[@]}"; do

              
        if [ "$CC_CLUSTER" = "beluga" ]; then
                  sbatch --export=ALL,\
BCKBN="$bckbn",\
DATA="$data",\
SEED="$id",\
LOSS="$loss",\
PRNM="$PRNM" \
./distributed_experiment_final.sh
              else
                  sbatch --export=ALL,\
BCKBN="$bckbn",\
DATA="$data",\
SEED="$id",\
LOSS="$loss",\
PRNM="$PRNM" \
./distributed_experiment_final.sh
              fi

      done
    done
  done
done

