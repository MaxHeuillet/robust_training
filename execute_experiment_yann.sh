#!/bin/bash

# Define variables
seeds=1

datas=( 
        'stanford_cars'
        'oxford-iiit-pet'
        'caltech101' 
        'flowers-102' 
        'fgvc-aircraft-2013b'
        'uc-merced-land-use-dataset' 
        ) 
 
losses=( 
  'CLASSIC_AT' 
  'TRADES_v2'
     )  

block_1=(
    "regnetx_004.pycls_in1k"
    'efficientnet-b0'
    'deit_tiny_patch16_224.fb_in1k'
    'mobilevit-small'
    'mobilenetv3_large_100.ra_in1k'
    'edgenext_small.usi_in1k'
    'coat_tiny.in1k'
)

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


echo "Submitting COMPUTING backbone jobs..."
submit_jobs "def-adurand" "${block_1[@]}" #TODO Define allocation