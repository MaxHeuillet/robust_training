#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm_clip_vith14.sh"

mkdir -p "$SCRIPT_DIR/logs/clip_vith14"

THREAT_MODELS=(linf4 linf8 linf30 l2_2 l2_8 l1_75 l1_300)

echo "Submitting 7 jobs for CLIP ViT-H/14 LAION-2B..."

for TM in "${THREAT_MODELS[@]}"; do
    JOB_ID=$(sbatch -J "$TM" "$JOB_SCRIPT" "$TM" | awk '{print $4}')
    echo "  $TM → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/clip_vith14/"