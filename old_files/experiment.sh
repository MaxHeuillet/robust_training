#!/bin/bash

#SBATCH --account=def-adurand

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=1

#SBATCH --mem-per-cpu=4000M
#SBATCH --time=04:00:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
module load arrow

source /home/mheuill/scratch/MYENV4/bin/activate



echo 'HZ: start python3 ./distributed_experiment.py ..at '; date

python3 ./distributed_experiment.py --data ${DATA} --model ${MODEL} --seed ${SEED} --n_rounds ${NROUNDS} --nb_epochs ${NBEPOCHS} --size ${SIZE} --active_strategy ${ASTRAT}  > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

