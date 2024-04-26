#!/bin/bash

#SBATCH --account=def-adurand

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=1

#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL


echo 'EVAL_TYPE' ${EVAL_TYPE} 

module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
# module load gurobi

source /home/mheuill/projects/def-adurand/mheuill/MYENV3/bin/activate
# source /home/mheuill/projects/def-adurand/mheuill/ENV_nogurobi/bin/activate
# source /home/mheuill/projects/def-adurand/mheuill/Gurobi_Py310/bin/activate

# virtualenv-clone /home/mheuill/projects/def-adurand/mheuill/MYENV3 $SLURM_TMPDIR/MYENV2
# deactivate
# source $SLURM_TMPDIR/MYENV3/bin/activate

echo 'HZ: start python3 ./experiment.py ..at '; date

python3 ./benchmark.py --eval_type ${EVAL_TYPE} --n_rounds ${NROUNDS} --nb_epochs ${NBEPOCHS} --round_size ${RSIZE} --active_strategy ${ASTRAT}  > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

