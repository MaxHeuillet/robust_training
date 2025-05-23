#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=11:58:55
#SBATCH --mail-type=ALL

source ./execute_setup.sh

# --- HPO Step ---
python ./distributed_experiment_final.py \
    --mode "train" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    > stdout_"$SLURM_JOB_ID" 2> stderr_"$SLURM_JOB_ID"

exit_code=$?
echo "HPO exit code: $exit_code"



If success, chain the next job
if [ $exit_code -eq 0 ]; then
    echo "HPO succeeded, submitting training job..."
    sbatch --account="$ACCOUNT" --mail-user="$EMAIL" \
       --export=ALL,ACCOUNT="$ACCOUNT",BCKBN="$BCKBN",DATA="$DATA",SEED="$SEED",LOSS="$LOSS",PRNM="$PRNM",EMAIL="$EMAIL" \
       ./job3_test_linf.sh
else
    echo "HPO failed. No further jobs will be submitted."
    exit 1
fi
