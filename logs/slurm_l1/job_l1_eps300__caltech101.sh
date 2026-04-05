#!/bin/bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=02:59:00
#SBATCH --job-name=craft_l1_eps300__caltech101
#SBATCH --mail-type=ALL
#SBATCH --output=/home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/slurm-%j__l1_eps300__caltech101.out
#SBATCH --error=/home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/slurm-%j__l1_eps300__caltech101.err

set -euo pipefail
source /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/execute_setup.sh

# HF token baked in at submission time from login node
export HF_TOKEN="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"
export HUGGING_FACE_HUB_TOKEN="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"

echo "[$(date)] Starting caltech101 L1 eps=300 on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Step 2 — run 4 shards in parallel, one per GPU
# CUDA_VISIBLE_DEVICES is set in the shell before Python starts
# so each process sees exactly one GPU as cuda:0
echo "[$(date)] Launching 4 shards..."
for SHARD_IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${SHARD_IDX} python /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/craft_shard.py         --dataset    caltech101         --norm       L1         --eps        300         --surrogate  clip_vith14         --batch_size 64         --shard_idx  ${SHARD_IDX}         --n_shards   4         > /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/l1_eps300__caltech101__shard${SHARD_IDX}.log 2>&1 &
done

echo "[$(date)] Waiting for all shards to complete..."
wait

echo "[$(date)] All shards done — merging and uploading"

# Step 3 — merge shards and upload to HuggingFace
python /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/craft_shard.py     --dataset   caltech101     --norm      L1     --eps       300     --surrogate clip_vith14     --merge     --upload_hf

echo "[$(date)] Done: caltech101 L1 eps=300"
