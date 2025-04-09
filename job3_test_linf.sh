#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:58:55
#SBATCH --mail-type=ALL

# Purge all loaded modules and load necessary ones
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a nccl/2.18.3 httpproxy
source ~/scratch/myenv_reprod/bin/activate


export PYTHONUNBUFFERED=1

# --- HPO Step ---
python ./distributed_experiment_final.py \
    --mode "test-Linf" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    > stdout_"$SLURM_JOB_ID" 2> stderr_"$SLURM_JOB_ID"

exit_code=$?
echo "HPO exit code: $exit_code"

# If success, chain the next job
if [ $exit_code -eq 0 ]; then
    echo "HPO succeeded, submitting training job..."
    sbatch --account="$ACCOUNT" --mail-user="$EMAIL" \
       --export=ALL,ACCOUNT="$ACCOUNT",BCKBN="$bckbn",DATA="$data",SEED="$id",LOSS="$loss",PRNM="$PRNM",EMAIL="$EMAIL" \
       ./job4_test_l1.sh
else
    echo "HPO failed. No further jobs will be submitted."
    exit 1
fi
