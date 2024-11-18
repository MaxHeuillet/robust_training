#!/bin/bash


#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:59:59
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2023 python/3.10 scipy-stack arrow httpproxy
source ~/scratch/MYENV4/bin/activate
# pip install  -r requirements.txt

if [ "${DATA}" = "Imagenet1k" ]; then
    echo 'unzip imagenet'
    mkdir -p $SLURM_TMPDIR/data
    tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data
    echo 'imagenet unzipped'
fi

echo 'HZ: start python3 ./distributed_experiment1.py ..at '; date


python3 ./distributed_experiment1.py \
    --init_lr ${LR} \
    --loss_function ${LOSS} \
    --sched ${SCHED} \
    --dataset ${DATA} \
    --seed ${SEED} \
    --iterations ${NITER} \
    --pruning_ratio ${RATIO} \
    --pruning_strategy ${PSTRAT} \
    --batch_strategy ${BSTRAT} \
    --aug ${AUG} \
    --backbone ${BCKBN} \
    --exp ${EXP} \
    > stdout_$SLURM_JOB_ID 2> stderr_$SLURM_JOB_ID