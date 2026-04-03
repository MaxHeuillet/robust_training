
# module --force purge
# module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a nccl/2.18.3 httpproxy

# python3.11 -m venv $SLURM_TMPDIR/myenv_reprod
# source $SLURM_TMPDIR/myenv_reprod/bin/activate
# pip install -r ./requirements.txt

# pip install "torch>=2.4" "torchvision>=0.20" --upgrade
# pip install "transformers>=4.49" "tokenizers>=0.21" "safetensors>=0.4.3" --upgrade

# export PYTHONUNBUFFERED=1

#!/bin/bash
# execute_setup.sh — Run on compute node (no internet required).
# Prerequisites: run predownload_models.sh on login node ONCE first.

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a nccl/2.18.3 httpproxy

python3.11 -m venv $SLURM_TMPDIR/myenv_reprod
source $SLURM_TMPDIR/myenv_reprod/bin/activate

# Install auto-attack from pre-cloned local copy (no internet needed)
# Pre-clone on login node: git clone https://github.com/fra31/auto-attack ~/links/scratch/auto-attack
pip install ~/links/scratch/auto-attack

# Install remaining requirements, skipping any git+ lines
grep -v 'git+' ./requirements.txt | pip install -r /dev/stdin

pip install "torch>=2.4" "torchvision>=0.20" --upgrade
pip install "transformers>=4.49" "tokenizers>=0.21" "safetensors>=0.4.3" --upgrade
pip install open_clip_torch

export PYTHONUNBUFFERED=1

# Point all model/data caches to scratch (pre-downloaded on login node)
export HF_HOME=~/links/scratch/robustgenbench/hf_cache
export HF_HUB_CACHE=~/links/scratch/robustgenbench/hf_cache
export TORCH_HOME=~/links/scratch/robustgenbench/model_cache
export HF_HUB_OFFLINE=1          # prevent any HF download attempts
export TRANSFORMERS_OFFLINE=1     # prevent any transformers download attempts