#!/bin/bash

#SBATCH --account=rrg-csubakan 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:20:59
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

# Purge all loaded modules and load necessary ones
module --force purge
module load StdEnv/2023 python/3.11 cuda scipy-stack arrow httpproxy
source ~/scratch/MYENV4/bin/activate

# Uncomment the following line if you need to install requirements
# pip install -r requirements.txt

echo "Processing dataset: ${DATA}"

# Create a temporary data directory
mkdir -p $SLURM_TMPDIR/data

# Define the path to the dataset archive
ARCHIVE_PATH=~/scratch/data/${DATA}

# Function to extract .tar.zst archives
extract_tar_zst() {
    local archive="${ARCHIVE_PATH}.tar.zst"
    tar -I zstd -xf "$archive" -C "$SLURM_TMPDIR/data"
}

# Function to extract .zip archives
extract_zip() {
    unzip "$HOME/scratch/data/archive.zip" -d "$SLURM_TMPDIR/data"
}

# Determine the extraction method based on the dataset
if [ "$DATA" == "stanford_cars" ]; then
    echo "Dataset is stanford_cars. Using unzip to extract."
    extract_zip
else
    echo "Dataset is not stanford_cars. Using tar with zstd to extract."
    extract_tar_zst
fi

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA} archive." >&2
    exit 1
fi

echo "Extraction of ${DATA} archive completed successfully at $(date)."

# Run the Python experiment script with appropriate arguments
python -u ./distributed_experiment2.py \
    --task "${TASK}" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --ft_type "${FTTYPE}" \
    --project_name "${PRNM}" \
    --exp "${EXP}" \
    > stdout_"$SLURM_JOB_ID" 2> stderr_"$SLURM_JOB_ID"
