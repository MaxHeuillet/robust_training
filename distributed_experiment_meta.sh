#!/bin/bash

seeds=$1

model='resnet50'
data=$2 #'Imagenet1k' #'CIFAR10'


sizes=( 1 2.5 10 ) 
strategies=('random' 'uncertainty' ) # 'attack_uncertainty' 'random' 'margin' 'entropy' 'attack'

# Second set of experiments
for size in "${sizes[@]}"; do
    for strategy in "${strategies[@]}"; do
        for ((id=0; id<$seeds; id++)); do
            sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,SIZE=$size,ASTRAT=$strategy,DATA=$data,MODEL=$model,SEED=$id ./distributed_experiment.sh
        done
    done
done



