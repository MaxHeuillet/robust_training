#!/bin/bash

# eval_types=( 'clean' 'PGD' 'AA') 
# for eval_type in "${eval_types[@]}"; do
#     echo 'eval_type' $eval_type 
#     sbatch --export=ALL,EVAL_TYPE=$eval_type,NROUNDS=$nrounds,NBEPOCHS=$nbepochs,RSIZE=$rsize,ASTRAT=$astrat ./experiment.sh     
#     done

# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='random' ./experiment.sh
# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='uncertainty' ./experiment.sh
# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='margin' ./experiment.sh
# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='entropy' ./experiment.sh
sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=6,NBEPOCHS=10,RSIZE=1000,ASTRAT='attack' ./experiment.sh

# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='random' ./experiment.sh
# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='uncertainty' ./experiment.sh
sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='margin' ./experiment.sh
# sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='entropy' ./experiment.sh
sbatch --export=ALL,EVAL_TYPE='clean',NROUNDS=1,NBEPOCHS=60,RSIZE=6000,ASTRAT='attack' ./experiment.sh