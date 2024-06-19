#!/bin/bash

# eval_types=( 'clean' 'PGD' 'AA') 
# for eval_type in "${eval_types[@]}"; do
#     echo 'eval_type' $eval_type 
#     sbatch --export=ALL,EVAL_TYPE=$eval_type,NROUNDS=$nrounds,NBEPOCHS=$nbepochs,RSIZE=$rsize,ASTRAT=$astrat ./experiment.sh     
#     done


seeds=$1

model='resnet50'
data='Imagenet1k' #'CIFAR10'

# First set of experiments
# for ((id=0; id<$seeds; id++)); do
#     sbatch --export=ALL,LOSS=$loss,NROUNDS=6,NBEPOCHS=10,SIZE=50000,ASTRAT='full',MODEL=$model,DATA=$data,SEED=$id ./experiment.sh
#     sbatch --export=ALL,LOSS=$loss,NROUNDS=1,NBEPOCHS=60,SIZE=50000,ASTRAT='full',MODEL=$model,DATA=$data,SEED=$id ./experiment.sh
# done

sizes=( 0.01 ) #500 2500 5000 7500 10000 12500 25000
strategies=('uncertainty') # 'attack_uncertainty' 'random' 'margin' 'entropy' 'attack'

# Second set of experiments
for size in "${sizes[@]}"; do
    for strategy in "${strategies[@]}"; do
        for ((id=0; id<$seeds; id++)); do
            #sbatch --export=ALL,LOSS=$loss,NROUNDS=6,NBEPOCHS=10,SIZE=$size,ASTRAT=$strategy,DATA=$data,MODEL=$model,SEED=$id ./experiment.sh
            sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,SIZE=$size,ASTRAT=$strategy,DATA=$data,MODEL=$model,SEED=$id ./distributed_experiment.sh
        done
    done
done



