#!/bin/bash


#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=03:00:00
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2023
module load python/3.10
module load scipy-stack
module load arrow

source ~/scratch/MYENV4/bin/activate
pip install  -r requirements.txt

nvidia-smi

pip list

if [ "${DATA}" = "Imagenet1k" ]; then
    echo 'unzip imagenet'
    mkdir -p $SLURM_TMPDIR/data
    tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data
    echo 'imagenet unzipped'
fi

echo 'HZ: start python3 ./distributed_training.py ..at '; date

echo "DATA = ${DATA}"
echo "MODEL = ${MODEL}"
echo "SEED = ${SEED}"
echo "NROUNDS = ${NROUNDS}"
echo "NBEPOCHS = ${NBEPOCHS}"
echo "SIZE = ${SIZE}"
echo "ACTIVE_STRATEGY = ${ASTRAT}"

python3 ./distributed_training.py --data ${DATA} --model ${MODEL} --seed ${SEED} --n_rounds ${NROUNDS} --nb_epochs ${NBEPOCHS} --size ${SIZE} --active_strategy ${ASTRAT} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID



