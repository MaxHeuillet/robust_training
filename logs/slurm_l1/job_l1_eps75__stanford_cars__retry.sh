#!/bin/bash
#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=02:59:00
#SBATCH --job-name=craft_l1_eps75__stanford_cars__retry
#SBATCH --mail-type=ALL
#SBATCH --output=/home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/slurm-%j__l1_eps75__stanford_cars__retry.out
#SBATCH --error=/home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/slurm-%j__l1_eps75__stanford_cars__retry.err

set -euo pipefail
source /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/execute_setup.sh

export HF_TOKEN="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"
export HUGGING_FACE_HUB_TOKEN="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"

echo "[$(date)] Starting stanford_cars L1 eps=75 on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Clean up any stale shard output from previous failed runs
# so shard_done.json flags don't incorrectly skip re-runs
echo "[$(date)] Cleaning stale shard dirs..."
rm -rf /tmp/robustgenbench/adversarial_examples/stanford_cars__zeroshot_clip_vith14_laion2b__l1_eps75_autoattack_standard__shard*
rm -rf /tmp/robustgenbench/work/stanford_cars

# Run 4 shards in parallel — lock serializes download + extraction
echo "[$(date)] Launching 4 shards..."
for SHARD_IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${SHARD_IDX} python /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/craft_shard.py         --dataset    stanford_cars         --norm       L1         --eps        75         --surrogate  clip_vith14         --batch_size 64         --shard_idx  ${SHARD_IDX}         --n_shards   4         > /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/logs/slurm_l1/l1_eps75__stanford_cars__shard${SHARD_IDX}.log 2>&1 &
done

echo "[$(date)] Waiting for all shards to complete..."
wait

echo "[$(date)] All shards done — merging and uploading"

python /home/m/mheuill/links/projects/aip-adurand/mheuill/robust_training/craft_shard.py     --dataset   stanford_cars     --norm      L1     --eps       75     --surrogate clip_vith14     --merge     --upload_hf

echo "[$(date)] Done: stanford_cars L1 eps=75"
