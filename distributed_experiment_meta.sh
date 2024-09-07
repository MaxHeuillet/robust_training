#!/bin/bash

seeds=$1

arch='resnet50'
data=$2  # 'Imagenet1k' or 'CIFAR10'
task=$3  # 'train' or 'evaluation'
loss=$4  # 'TRADES' 'TRADES_v2' 'Madry'
sched=$5  # 'sched' or 'nosched'
iterations=$6

init_lrs=( 0.2 0.01 0.001 )  # 
pruning_ratios=( 0.7 )  #0.3 0.5  
pruning_strategies=(   'TS_pruning' 'random' 'uncertainty' 'score_v1' 'score_v2'  ) # 'TS_context'
batch_strategies=('random')

# Second set of experiments
for pruning_ratio in "${pruning_ratios[@]}"; do
    for pruning_strategy in "${pruning_strategies[@]}"; do
        for batch_strategy in "${batch_strategies[@]}"; do
            for init_lr in "${init_lrs[@]}"; do
                for ((id=0; id<$seeds; id++)); do
                    if [ "$CC_CLUSTER" = "cedar" ]; then
                        sbatch --export=ALL,\
NITER=$iterations,\
TASK=$task,\
RATIO=$pruning_ratio,\
PSTRAT=$pruning_strategy,\
BSTRAT=$batch_strategy,\
DATA=$data,\
ARCH=$arch,\
SEED=$id,\
LOSS=$loss,\
SCHED=$sched,\
LR=$init_lr \
./distributed_experiment_cedar.sh
                    else
                        sbatch --export=ALL,\
NITER=$iterations,\
TASK=$task,\
RATIO=$pruning_ratio,\
PSTRAT=$pruning_strategy,\
BSTRAT=$batch_strategy,\
DATA=$data,\
ARCH=$arch,\
SEED=$id,\
LOSS=$loss,\
SCHED=$sched,\
LR=$init_lr \
./distributed_experiment_other.sh
                    fi
                done
            done
        done
    done
done
