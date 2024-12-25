#!/bin/bash

#SBATCH --account=rrg-csubakan 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:58:59
#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2023 python/3.11 cuda scipy-stack arrow httpproxy
source ~/scratch/MYENV4/bin/activate
# pip install  -r requirements.txt


echo "Processing dataset: ${DATA}"

mkdir -p $SLURM_TMPDIR/data

tar -I zstd -xf ~/scratch/data/${DATA}.tar.zst -C $SLURM_TMPDIR/data
    
# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA}.tar.zst" >&2
    exit 1
fi
    echo "Extraction of ${DATA}.tar.zst completed successfully at $(date)."


echo 'HZ: start python3 ./distributed_experiment1.py ..at '; date


python3 ./distributed_experiment2.py \
    --task ${TASK} \
    --loss_function ${LOSS} \
    --dataset ${DATA} \
    --seed ${SEED} \
    --backbone ${BCKBN} \
    --ft_type ${FTTYPE} \
    --project_name ${PRNM} \
    --exp ${EXP} \
    > stdout_$SLURM_JOB_ID 2> stderr_$SLURM_JOB_ID




# mv ~/scratch/data/* "$SLURM_TMPDIR/data/"


# if [ "${DATA}" = "Imagenet1k" ]; then
#     echo 'unzip imagenet'
#     mkdir -p $SLURM_TMPDIR/data
#     tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data
#     echo 'imagenet unzipped'
# fi
