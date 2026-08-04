#!/bin/bash
cd /project/6102313/mheuill/robust_training

BASE_RETRY_IDS="390428 390429 390430 390431 390432 390433 390434 390435 390436 390437 390438 390439 390440 390441 390442 390443 390444 390445 390446 390447 390448 390449"
declare -A RESOLVED
declare -A IS_JOB3
PENDING_SET=()
for j in $BASE_RETRY_IDS; do PENDING_SET+=("$j"); done

DEADLINE=$(( $(date +%s) + 72000 ))  # 20h safety cap

log() { echo "[$(date '+%F %T')] $*"; }

log "Tracking Base retry chain: 22 job2 IDs + any chained job3 evals."

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STILL_PENDING=()
  for j in "${PENDING_SET[@]}"; do
    if [ -n "${RESOLVED[$j]}" ]; then continue; fi

    state=$(sacct -j "$j" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$state" ]; then
      STILL_PENDING+=("$j")
      continue
    fi

    case "$state" in
      COMPLETED)
        if [ -n "${IS_JOB3[$j]}" ]; then
          RESOLVED[$j]=1
          log "job3 (eval) $j COMPLETED."
        else
          out="logs/slurm-${j}.out"
          chained=$(grep -oE "Submitted batch job [0-9]+" "$out" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
          if [ -n "$chained" ]; then
            log "job2 $j COMPLETED, chained job3=$chained. Now tracking $chained."
            IS_JOB3[$chained]=1
            PENDING_SET+=("$chained")
            RESOLVED[$j]=1
          else
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
    log "Full Base retry chain finished (all job2 + chained job3 terminal)."
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
  log "Released $(echo "$HELD" | wc -l) jobs."
else
  log "No held Small/Tiny jobs found (already released or none pending)."
fi
log "Done."
