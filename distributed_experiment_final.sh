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
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a httpproxy
source ~/scratch/myenv_reprod/bin/activate

# export LD_PRELOAD="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.10.0/lib64/libopencv_cudaimgproc.so.410:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.10.0/lib64/libopencv_cudafeatures2d.so.410"

export PYTHONUNBUFFERED=1

bash ./dataset_to_tmdir.sh "$DATA"

# Run the Python experiment script with appropriate arguments
python ./distributed_experiment_final.py \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    --exp "${EXP}" \
    > stdout_"$SLURM_JOB_ID" 2> stderr_"$SLURM_JOB_ID"
