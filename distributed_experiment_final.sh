#!/bin/bash

#SBATCH --account=rrg-csubakan
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=11:58:55
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

# Purge all loaded modules and load necessary ones
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.12 arrow/18.1.0 scipy-stack/2024a httpproxy
source ~/scratch/MYENV4/bin/activate

# export LD_PRELOAD="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.10.0/lib64/libopencv_cudaimgproc.so.410:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.10.0/lib64/libopencv_cudafeatures2d.so.410"

export PYTHONUNBUFFERED=1

echo "Processing dataset: ${DATA}"

# Create a temporary data directory
mkdir -p $SLURM_TMPDIR/data

# Define the path to the dataset archive (without extension)
ARCHIVE_PATH=~/scratch/data/${DATA}

# Function to extract .tar.zst archives
extract_tar_zst() {
    local archive="${ARCHIVE_PATH}.tar.zst"
    echo "Extracting ${archive} with tar + zstd..."
    tar -I zstd -xf "$archive" -C "$SLURM_TMPDIR/data"
}

# Function to extract .zip archives
extract_zip() {
    local archive="${ARCHIVE_PATH}.zip"
    echo "Extracting ${archive} with unzip..."
    unzip -q "$archive" -d "$SLURM_TMPDIR/data"
}

# If your dataset is zipped, add its name to the OR list below.
if [ "$DATA" == "stanford_cars" ] || \
   [ "$DATA" == "uc-merced-land-use-dataset" ] || \
   [ "$DATA" == "kvasir-dataset" ]; then
    echo "Dataset is $DATA. Using unzip to extract."
    extract_zip
else
    echo "Dataset is not in the zip list. Using tar.zst to extract."
    extract_tar_zst
fi

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA} archive." >&2
    exit 1
fi

echo "Extraction of ${DATA} archive completed successfully at $(date)."

# Run the Python experiment script with appropriate arguments
python ./distributed_experiment_final.py \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    --exp "${EXP}" \
    > stdout_"$SLURM_JOB_ID" 2> stderr_"$SLURM_JOB_ID"
