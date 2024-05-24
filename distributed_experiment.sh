#!/bin/bash

#SBATCH --account=def-adurand

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=1

#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:30:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
# module load gurobi

source /home/mheuill/scratch/MYENV4/bin/activate
# source /home/mheuill/projects/def-adurand/mheuill/MYENV3/bin/activate
# source /home/mheuill/projects/def-adurand/mheuill/ENV_nogurobi/bin/activate
# source /home/mheuill/projects/def-adurand/mheuill/Gurobi_Py310/bin/activate

# virtualenv-clone /home/mheuill/projects/def-adurand/mheuill/MYENV3 $SLURM_TMPDIR/MYENV2
# deactivate
# source $SLURM_TMPDIR/MYENV3/bin/activate

echo 'HZ: start python3 ./experiment.py ..at '; date


python3 ./distributed_training.py > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

