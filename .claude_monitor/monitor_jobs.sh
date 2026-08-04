#!/bin/bash
cd /project/6102313/mheuill/robust_training
JOBS="389679 389680 389681 389682 389683"
DEADLINE=$(( $(date +%s) + 900 ))  # 15 min safety cap

echo "Monitoring jobs: $JOBS"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  ALL_DONE=1
  for j in $JOBS; do
    f="stderr_train_${j}"
    out="logs/slurm-${j}.out"
    if [ -f "$f" ] && grep -q "named symbol not found\|Traceback" "$f" 2>/dev/null; then
      echo "JOB $j: FAILED (crash signature found in $f)"
      continue
    fi
    if [ -f "$out" ] && grep -q "Training exit code: 0" "$out" 2>/dev/null; then
      echo "JOB $j: TRAINING SUCCEEDED (exit code 0)"
      continue
    fi
    if [ -f "$out" ] && grep -q "Training exit code:" "$out" 2>/dev/null; then
      code=$(grep "Training exit code:" "$out" | tail -1)
      echo "JOB $j: FINISHED - $code"
      continue
    fi
    state=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
    if [ -z "$state" ]; then
      echo "JOB $j: no longer in queue, no exit-code line found yet (check sacct)"
      continue
    fi
    ALL_DONE=0
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "All jobs reached a terminal state."
    break
  fi
  sleep 20
done
echo "=== Monitor loop finished ==="
for j in $JOBS; do
  echo "--- sacct $j ---"
  sacct -j "$j" --format=JobID,State,ExitCode,Elapsed 2>&1 | head -3
done
