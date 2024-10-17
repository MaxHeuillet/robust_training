#!/bin/bash

# Define variables
seeds=1
archs=( 'convnext' ) # 'resnet50' 'vitsmall'
datas=( 'CIFAR100', ) # 'Aircraft' 'EuroSAT',  'CIFAR10',
task='train'
losses=('TRADES_v2', 'APGD')
sched='nosched'
iterations=50
aug='aug'
exp='RQ1'

init_lrs=( 0.0001 0.0005 0.001  )
pruning_ratios=( 0 )
pruning_strategies=( 'random' )
batch_strategies=('random')
pts=('imagenet21k_non_robust', 'imagenet1k_non_robust', 'imagenet1k_robust')

# Loop over architectures, pruning ratios, strategies, and learning rates
for data in "${datas[@]}"; do
  for loss in "${losses[@]}"; do
    for arch in "${archs[@]}"; do
      for pruning_ratio in "${pruning_ratios[@]}"; do
        for pruning_strategy in "${pruning_strategies[@]}"; do
          for batch_strategy in "${batch_strategies[@]}"; do
            for init_lr in "${init_lrs[@]}"; do
              for id in $(seq 1 $seeds); do
                for pt in "${pts[@]}"; do

                  if [ "$CC_CLUSTER" = "beluga" ]; then
          
              ### Pretrained non robust
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
PRETRAINED=$pt,\
LORA='nolora',\
EXP=$exp \
./distributed_experiment_beluga.sh

                  else

              ### Pretrained non robust
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
PRETRAINED=$pt,\
LORA='nolora',\
EXP=$exp \
./distributed_experiment_other.sh

                  fi

                done
              done
            done
          done
        done
      done
    done
  done
done