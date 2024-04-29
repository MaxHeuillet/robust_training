#!/bin/bash

# eval_types=( 'clean' 'PGD' 'AA') 
# for eval_type in "${eval_types[@]}"; do
#     echo 'eval_type' $eval_type 
#     sbatch --export=ALL,EVAL_TYPE=$eval_type,NROUNDS=$nrounds,NBEPOCHS=$nbepochs,RSIZE=$rsize,ASTRAT=$astrat ./experiment.sh     
#     done

seeds=$1
model='resnet50'
data='cifar10'
for ((id=0; id<$seeds; id+=1)); do

    sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=0,ASTRAT='full',MODEL=$model,DATA=$data,SEED=$id ./experiment.sh

    # sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='random',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='uncertainty',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='margin',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='entropy',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='attack',SEED=$id ./experiment.sh

    sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=0,ASTRAT='full',MODEL=$model,DATA=$data,SEED=$id ./experiment.sh

    # sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='random',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='uncertainty',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='margin',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='entropy',SEED=$id ./experiment.sh
    # sbatch --export=ALL,NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='attack',SEED=$id ./experiment.sh

    done

