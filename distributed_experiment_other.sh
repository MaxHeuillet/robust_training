#!/bin/bash

#SBATCH --account=rrg-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=03:00:00
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2023 python/3.10 scipy-stack httpproxy arrow

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
    --_init_lr ${LR} \
    --loss_function ${LOSS} \
    --sched ${SCHED} \
    --task ${TASK} \
    --dataset ${DATA} \
    --arch ${ARCH} \
    --seed ${SEED} \
    --iterations ${NITER} \
    --pruning_ratio ${RATIO} \
    --pruning_strategy ${PSTRAT} \
    --batch_strategy ${BSTRAT} \
    > stdout_$SLURM_JOB_ID 2> stderr_$SLURM_JOB_ID






# python3 ./distributed_experiment1.py --_init_lr 0.01 --loss_function 'TRADES' --sched 'sched' --task 'train' --dataset 'CIFAR10' --arch 'resnet50' --seed 0 --iterations 2 --pruning_ratio 0.5 --pruning_strategy 'random' --batch_strategy 'random' > stdout_$SLURM_JOB_ID 2> stderr_$SLURM_JOB_ID
