#!/bin/bash

# Define variables
seeds=1
datas=( 'Imagenette'  ) # 'Flowers' 'Aircraft' 'CIFAR10' 'CIFAR100'  'EuroSAT' 
losses=(  'TRADES_v2' ) #'CLASSIC_AT'

backbones=(
  'convnext_tiny' #'robust_convnext_tiny' 'convnext_tiny.fb_in22k' # 'random_convnext_tiny'
) 

ft_type=( 'full_fine_tuning' )

tasks=( 'HPO' 'train'  ) # 'test' 'dormant'

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
