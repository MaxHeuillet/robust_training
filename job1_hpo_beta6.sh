#!/bin/bash
# =============================================================================
# job1_hpo_beta6.sh
# Genuine HPO search (lr1, lr2, weight_decay1, weight_decay2, scheduler) for
# TRADES beta=6, fixing the fairness issue where beta=6 runs reused
# beta=1-tuned hyperparameters. Fixed compute budget of 140 minutes per
# configuration (see utils/hp_opt.py), matching the paper's stated protocol.
#
# On success, chains to job2_train_multiseed.sh (not the older job2_train.sh
# that job1_hpo.sh chains to) so training picks up the H100 batch-size /
# race-condition fixes made earlier this session, and does NOT auto-chain to
# the old white-box eval scripts -- white-box eval is launched manually once
# training completes, same as every other combo this session.
#
# Required env vars (passed via --export):
#   ACCOUNT, BCKBN, DATA, SEED, LOSS, PRNM, EMAIL
#
# IMPORTANT: PRNM must be a NEW, distinct project name (e.g.
# "{backbone}_TRADES_beta6_hpo") for both the HPO output directory
# (configs/HPO_results/{PRNM}/) and the trained checkpoint location
# (trained_statedicts/{PRNM}/) -- do NOT reuse the existing
# "{backbone}_TRADES_beta6" project name, which would overwrite the
# already-trained (reused-hyperparameter) checkpoints and results.
# =============================================================================

#SBATCH --account=aip-adurand
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=02:59:00
#SBATCH --mail-type=ALL
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

source ./execute_setup.sh
source ./setup_paths.sh

echo "============================================"
echo "  JOB: HPO search for TRADES beta=6"
echo "  Backbone : ${BCKBN}"
echo "  Dataset  : ${DATA}"
echo "  Project  : ${PRNM}"
echo "  Node     : $(hostname)"
echo "============================================"

# --- Download clean test/train data from HuggingFace (mirrors job2_train_multiseed.sh) ---
echo "Downloading training data for ${DATA}..."
python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

dataset   = os.environ["DATA"]
hf_repo   = "legolasflagstaff/RobustGenBench"
data_dir  = os.path.join(os.environ.get("SLURM_TMPDIR", "/tmp"), "datasets")
archive   = f"{dataset}_processed.tar.zst"
dest_path = os.path.join(data_dir, archive)

if os.path.exists(dest_path):
    print(f"  Archive already present: {dest_path}")
    sys.exit(0)

os.makedirs(data_dir, exist_ok=True)
print(f"  Downloading {archive} from {hf_repo} ...")
hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=archive, local_dir=data_dir)
print(f"  Done. Archive at {dest_path}")
PYEOF

dl_exit=$?
if [ $dl_exit -ne 0 ]; then
    echo "ERROR: Training data download failed (exit ${dl_exit}). Aborting."
    exit 1
fi

# --- Download backbone weights if not already present (mirrors job2_train_multiseed.sh) ---
echo "Checking backbone weights..."
python - <<'PYEOF'
import os, timm, torch
from pathlib import Path

backbone = os.environ["BCKBN"]
dest_dir = Path(os.path.expanduser("~/links/scratch/mheuill/my_backbones/"))
dest_dir.mkdir(parents=True, exist_ok=True)
dest_path = dest_dir / f"{backbone}.pt"

if dest_path.exists():
    print(f"  Backbone weights already exist: {dest_path}")
else:
    print(f"  Downloading {backbone} from timm...")
    model = timm.create_model(backbone, pretrained=True)
    torch.save(model.state_dict(), dest_path)
    print(f"  Saved to {dest_path}")
PYEOF

echo "Starting HPO search..."
python ./distributed_experiment_final.py \
    --mode "hpo" \
    --loss_function "${LOSS}" \
    --dataset "${DATA}" \
    --seed "${SEED}" \
    --backbone "${BCKBN}" \
    --project_name "${PRNM}" \
    --beta 6.0 \
    > stdout_hpo_"$SLURM_JOB_ID" 2> stderr_hpo_"$SLURM_JOB_ID"

exit_code=$?
echo "HPO exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "HPO succeeded, submitting training job..."
    sbatch --account="$ACCOUNT" --mail-user="$EMAIL" \
       --export=ALL,ACCOUNT="$ACCOUNT",BCKBN="$BCKBN",DATA="$DATA",SEED="$SEED",LOSS="$LOSS",PRNM="$PRNM",HPO_SOURCE_PRNM="$PRNM",EMAIL="$EMAIL" \
       ./job2_train_multiseed.sh
else
    echo "HPO failed. No further jobs will be submitted."
    exit 1
fi
