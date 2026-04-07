#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm_siglip2_so400m_sharded.sh"

mkdir -p "$SCRIPT_DIR/logs/siglip2_so400m"

THREAT_MODELS=(linf8 linf30 l2_2 l2_8 l1_75 l1_300)

echo "Submitting 6 sharded jobs for SigLIP2-SO400M-NaFlex..."
echo "  (4 GPUs per dataset, datasets sequential within each job)"
echo ""

for TM in "${THREAT_MODELS[@]}"; do
    JOB_ID=$(sbatch -J "$TM" "$JOB_SCRIPT" "$TM" | awk '{print $4}')
    echo "  $TM → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/siglip2_so400m/"