#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm_siglip2_so400m_384.sh"

mkdir -p "$SCRIPT_DIR/logs/siglip2_so400m_384"

THREAT_MODELS=(linf8 l2_2 l2_8 l1_75 l1_300)

echo "Submitting 5 jobs for SigLIP2 SO400M patch14-384..."

for TM in "${THREAT_MODELS[@]}"; do
    JOB_ID=$(sbatch -J "$TM" "$JOB_SCRIPT" "$TM" | awk '{print $4}')
    echo "  $TM → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/siglip2_so400m_384/"