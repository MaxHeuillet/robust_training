#!/bin/bash
cd /project/6102313/mheuill/robust_training
JOB=389735
DEADLINE=$(( $(date +%s) + 10800 ))  # up to 3h (job's own time limit)

echo "Monitoring test job $JOB"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  out="logs/slurm-${JOB}.out"
  serr="stderr_test_${JOB}"

  if [ -f "$out" ] && grep -q "Test exit code: 0" "$out" 2>/dev/null; then
    echo "JOB $JOB: TEST SUCCEEDED (exit code 0)"
    echo "--- tail of $out ---"
    tail -20 "$out"
    exit 0
  fi

  if [ -f "$out" ] && grep -q "Test exit code:" "$out" 2>/dev/null; then
    code=$(grep "Test exit code:" "$out" | tail -1)
    echo "JOB $JOB: FAILED - $code"
    echo "--- tail of $serr ---"
    tail -60 "$serr" 2>/dev/null
    exit 1
  fi

  if [ -f "$serr" ] && grep -qE "Traceback|Error|CUDA out of memory" "$serr" 2>/dev/null; then
    echo "JOB $JOB: error signature detected in $serr (job may still be running / retrying)"
    tail -60 "$serr"
  fi

  state=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
  if [ -z "$state" ]; then
    echo "JOB $JOB: left the queue with no 'Test exit code' line yet - checking sacct"
    sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed 2>&1
    echo "--- tail of slurm err ---"
    tail -60 "logs/slurm-${JOB}.err" 2>/dev/null
    echo "--- tail of stderr_test ---"
    tail -60 "$serr" 2>/dev/null
    exit 2
  fi

  sleep 30
done
echo "=== Monitor deadline reached, job $JOB still running ==="
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed 2>&1
