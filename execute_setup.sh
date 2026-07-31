module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a nccl/2.18.3 httpproxy

python3.11 -m venv $SLURM_TMPDIR/myenv_reprod
source $SLURM_TMPDIR/myenv_reprod/bin/activate
pip install -r ./requirements.txt

pip install "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"
pip install "transformers>=4.49" "tokenizers>=0.21" "safetensors>=0.4.3" --upgrade

export PYTHONUNBUFFERED=1