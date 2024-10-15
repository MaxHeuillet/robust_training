#!/bin/bash

# Define variables
seeds=1
archs=( 'convnext' ) # 'resnet50' 'vitsmall'
data='CIFAR10'
task='train'
loss='APGD'
sched='nosched'
iterations=50
aug='aug'
exp='RQ2'

init_lrs=( 0.001 )
pruning_ratios=( 0.7 0.0 0.3 0.5  )
pruning_strategies=( 'fixed_stratified' )
batch_strategies=('random')

# Loop over architectures, pruning ratios, strategies, and learning rates
for arch in "${archs[@]}"; do
  for pruning_ratio in "${pruning_ratios[@]}"; do
    for pruning_strategy in "${pruning_strategies[@]}"; do
      for batch_strategy in "${batch_strategies[@]}"; do
        for init_lr in "${init_lrs[@]}"; do
          for id in $(seq 1 $seeds); do
          
            ### Pretrained robust
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
LR=$init_lr,\
AUG=$aug,\
PRETRAINED='imagenet1k_non_robust',\
LORA='nolora',\
EXP=$exp \
./distributed_experiment_other.sh

          done
        done
      done
    done
  done
done
