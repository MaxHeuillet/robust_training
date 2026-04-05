#!/usr/bin/env bash
# ============================================================
# submit_all_metaclip_sharded.sh
# Submit all 6 threat models as independent SLURM jobs.
# Each job uses 4 GPUs per dataset (sharded), datasets sequential.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm_metaclip_sharded.sh"

mkdir -p "$SCRIPT_DIR/logs/metaclip_h14"

THREAT_MODELS=(linf8 linf30 l2_2 l2_8 l1_75 l1_300)

echo "Submitting 6 sharded jobs for MetaCLIP ViT-H/14 fullcc2.5b..."
echo "  (4 GPUs per dataset, datasets sequential within each job)"
echo ""

for TM in "${THREAT_MODELS[@]}"; do
    JOB_ID=$(sbatch -J "$TM" "$JOB_SCRIPT" "$TM" | awk '{print $4}')
    echo "  $TM → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/metaclip_h14/"