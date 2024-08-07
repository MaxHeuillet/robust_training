#!/bin/bash

seeds=$1

model='resnet50'
data=$2 #'Imagenet1k' #'CIFAR10'
task=$3 #'train' 'evaluation'
loss=$4 #'TRADES' 'Madry'
sched=$5 #'sched' 'nosched'

lrs=( 0.01 0.001 0.0001 )  
sizes=( 5 10 25 )  
strategies=('random' ) #'uncertainty' 'attack_uncertainty' 'random' 'margin' 'entropy' 'attack'

# Second set of experiments
for size in "${sizes[@]}"; do
    for strategy in "${strategies[@]}"; do
        for lr in "${lrs[@]}"; do
            for ((id=0; id<$seeds; id++)); do
                if [ "$CC_CLUSTER" = "cedar" ]; then
                    sbatch --export=ALL,NROUNDS=1,NBEPOCHS=100,TASK=$task,SIZE=$size,ASTRAT=$strategy,DATA=$data,MODEL=$model,SEED=$id,LOSS=$loss,SCHED=$sched,LR=$lr ./distributed_experiment_cedar.sh
                else
                    sbatch --export=ALL,NROUNDS=1,NBEPOCHS=100,TASK=$task,SIZE=$size,ASTRAT=$strategy,DATA=$data,MODEL=$model,SEED=$id,LOSS=$loss,SCHED=$sched,LR=$lr ./distributed_experiment_other.sh
                fi
            done
        done
    done
done



