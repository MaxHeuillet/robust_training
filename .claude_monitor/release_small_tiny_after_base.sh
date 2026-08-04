#!/bin/bash
cd /project/6102313/mheuill/robust_training

BASE_JOB2_IDS=$(seq 389679 389732)
declare -A RESOLVED   # job_id -> 1 once terminal
declare -A IS_JOB3    # job_id -> 1 if it's a chained eval job
PENDING_SET=()
for j in $BASE_JOB2_IDS; do PENDING_SET+=("$j"); done

DEADLINE=$(( $(date +%s) + 72000 ))  # 20h safety cap

log() { echo "[$(date '+%F %T')] $*"; }

log "Tracking Base job2 chain: 389679-389732 (54 jobs) + any chained job3 evals."

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STILL_PENDING=()
  for j in "${PENDING_SET[@]}"; do
    if [ -n "${RESOLVED[$j]}" ]; then continue; fi

    state=$(sacct -j "$j" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$state" ]; then
      # not yet in accounting DB
      STILL_PENDING+=("$j")
      continue
    fi

    case "$state" in
      COMPLETED)
        if [ -n "${IS_JOB3[$j]}" ]; then
          RESOLVED[$j]=1
          log "job3 (eval) $j COMPLETED."
        else
          # job2: check if it chained a job3, pick up its ID from the slurm out log
          out="logs/slurm-${j}.out"
          chained=$(grep -oE "Submitting test job|Submitted batch job [0-9]+" "$out" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
          if [ -n "$chained" ]; then
            log "job2 $j COMPLETED, chained job3=$chained. Now tracking $chained."
            IS_JOB3[$chained]=1
            PENDING_SET+=("$chained")
            RESOLVED[$j]=1
          else
            # completed but no chain found yet - log line may not be flushed, retry
            STILL_PENDING+=("$j")
          fi
        fi
        ;;
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|DEADLINE|BOOT_FAIL)
        RESOLVED[$j]=1
        kind="job2"; [ -n "${IS_JOB3[$j]}" ] && kind="job3"
        log "$kind $j terminal with state=$state (no further chain)."
        ;;
      *)
        STILL_PENDING+=("$j")
        ;;
    esac
  done

  PENDING_SET=("${STILL_PENDING[@]}")

  if [ "${#PENDING_SET[@]}" -eq 0 ]; then
    log "Full Base chain finished (all job2 + chained job3 terminal)."
    break
  fi

  log "Still waiting on ${#PENDING_SET[@]} job(s): ${PENDING_SET[*]}"
  sleep 60
done

if [ "${#PENDING_SET[@]}" -ne 0 ]; then
  log "WARNING: deadline reached with jobs still unresolved: ${PENDING_SET[*]}. NOT releasing Small/Tiny yet."
  exit 1
fi

log "Releasing all held Small/Tiny jobs (389741-389848)..."
HELD=$(squeue -u "$USER" -h -t PENDING -o "%i %r" | awk '$1>=389741 && $1<=389848 && $2=="JobHeldUser" {print $1}')
if [ -n "$HELD" ]; then
  scontrol release $HELD
  log "Released: $HELD"
else
  log "No held Small/Tiny jobs found (already released or none pending)."
fi
log "Done."
