#!/bin/bash
# =============================================================================
# job_common_fill_all.sh
# Fills in the missing common_acc metric for many (backbone, dataset, seed,
# loss) combos within a SINGLE SLURM allocation, instead of one job per combo.
# Avoids 192x queueing/allocation wait and 192x environment setup; datasets
# extracted once per job (see the metadata.json skip-check added to
# utils/movers.py) are reused across every combo that shares that dataset.
#
# Required env vars:
#   ACCOUNT, EMAIL
#   COMBO_LIST  - path to a CSV file, one combo per line, no header:
#                 bckbn,data,seed,loss,prnm,hpo_source_prnm
#
# Ordered by dataset in the combo list to maximize extraction-skip reuse.
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
echo "  JOB: common-corruption fill-in (multi-combo, single allocation)"
echo "  Combo list : ${COMBO_LIST}"
echo "  Node       : $(hostname)"
echo "  GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

total=$(wc -l < "${COMBO_LIST}")
echo "Total combos: ${total}"

n_ok=0
n_fail=0
i=0

while IFS=',' read -r bckbn data seed loss prnm hpo_src; do
    i=$((i+1))
    echo ""
    echo "=== [$i/$total] backbone=${bckbn} dataset=${data} seed=${seed} loss=${loss} project=${prnm} ==="

    # --- Download clean test data if not already present (idempotent) ---
    DATA="${data}" python - <<'PYEOF'
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

    # --- Download backbone weights if not already present (idempotent) ---
    BCKBN="${bckbn}" python - <<'PYEOF'
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

    python ./distributed_experiment_final.py \
        --mode "test-common-only" \
        --loss_function "${loss}" \
        --dataset "${data}" \
        --seed "${seed}" \
        --backbone "${bckbn}" \
        --project_name "${prnm}" \
        --hpo_source_project "${hpo_src}" \
        > "stdout_common_${SLURM_JOB_ID}_combo${i}" 2> "stderr_common_${SLURM_JOB_ID}_combo${i}"

    exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "  OK"
        n_ok=$((n_ok+1))
    else
        echo "  FAILED (exit ${exit_code}) - see stderr_common_${SLURM_JOB_ID}_combo${i}"
        n_fail=$((n_fail+1))
    fi
done < "${COMBO_LIST}"

echo ""
echo "============================================"
echo "  DONE: ${n_ok}/${total} succeeded, ${n_fail}/${total} failed"
echo "============================================"

if [ ${n_fail} -gt 0 ]; then
    exit 1
fi
