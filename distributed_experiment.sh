#!/bin/bash

#SBATCH --account=def-adurand

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=4

#SBATCH --mem-per-cpu=4000M
#SBATCH --time=03:00:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL



module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
module load arrow

source ~/scratch/MYENV4/bin/activate

pip list

if [ "${DATA}" = "Imagenet1k" ]; then

    echo 'unzip imagenet';

    mkdir -p $SLURM_TMPDIR/data
    tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data

    echo 'imagenet unzipped';
fi

echo 'HZ: start python3 ./distributed_training.py ..at '; date


python3 ./distributed_training.py > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

