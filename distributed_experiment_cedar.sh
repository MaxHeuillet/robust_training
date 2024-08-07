#!/bin/bash


#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=12:00:00
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2023
module load python/3.10
module load scipy-stack
module load arrow
module load httpproxy

source ~/scratch/MYENV4/bin/activate
pip install  -r requirements.txt

# wandb login --relogin <<EOF
# 12c5bc9e9809f53b1150856be2ae3614c88e4639
# EOF

if [ "${DATA}" = "Imagenet1k" ]; then
    echo 'unzip imagenet'
    mkdir -p $SLURM_TMPDIR/data
    tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data
    echo 'imagenet unzipped'
fi

echo 'HZ: start python3 ./distributed_experiment1.py ..at '; date

echo "DATA = ${DATA}"
echo "MODEL = ${MODEL}"
echo "SEED = ${SEED}"
echo "NROUNDS = ${NROUNDS}"
echo "NBEPOCHS = ${NBEPOCHS}"
echo "SIZE = ${SIZE}"
echo "ACTIVE_STRATEGY = ${ASTRAT}"
echo "TASK = ${TASK}"
echo "LOSS = ${LOSS}"
echo "SCHED = ${SCHED}"
echo "LR = ${LR}"

python3 ./distributed_experiment1.py --lr ${LR} --loss ${LOSS} --sched ${SCHED} --task ${TASK} --data ${DATA} --model ${MODEL} --seed ${SEED} --n_rounds ${NROUNDS} --nb_epochs ${NBEPOCHS} --size ${SIZE} --active_strategy ${ASTRAT} --slurm_job_id ${SLURM_JOB_ID} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID


# wandb sync wandb/offline-run-*