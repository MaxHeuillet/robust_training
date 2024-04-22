#!/bin/bash

eval_types=( 'clean' 'PGD' 'AA') 

for eval_type in "${eval_types[@]}"; do

    echo 'eval_type' $eval_type 
    sbatch --export=ALL,EVAL_TYPE=$eval_type ./benchmark_launch.sh     

    done
    
done