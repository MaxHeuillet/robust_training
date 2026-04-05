#!/usr/bin/env bash
# ============================================================
# submit_all_metaclip_h14.sh
# Submit all 6 threat models as independent SLURM jobs.
# Each job gets 4 GPUs and processes all 6 datasets.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm_metaclip_h14_job.sh"

mkdir -p "$SCRIPT_DIR/logs/metaclip_h14"

THREAT_MODELS=(linf8 linf30 l2_2 l2_8 l1_75 l1_300)

echo "Submitting 6 jobs for MetaCLIP ViT-H/14 fullcc2.5b..."
echo ""

for TM in "${THREAT_MODELS[@]}"; do
    JOB_ID=$(sbatch -J "$TM" "$JOB_SCRIPT" "$TM" | awk '{print $4}')
    echo "  $TM → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/metaclip_h14/"