#!/bin/bash
# =============================================================================
# rorqual_setup.sh
# Run this from the repo root on rorqual (after `git clone` + `git pull`).
# Pulls the beta=6 HPO checkpoint tarball from TAMIA and extracts it, so the
# 40 combos listed in rorqual_ready_combos.csv can be white-box evaluated
# here instead of waiting in TAMIA's queue.
#
# Usage: bash rorqual_setup.sh <tamia_username>
# =============================================================================

set -e

TAMIA_USER="${1:?Usage: bash rorqual_setup.sh <tamia_username>}"
TARBALL_REMOTE="/home/m/mheuill/links/scratch/mheuill/robust_training/beta6_hpo_checkpoints_for_rorqual.tar"
TARBALL_LOCAL="beta6_hpo_checkpoints_for_rorqual.tar"
EXPECTED_MD5="b0f6e620a68c2ce698810547a1774e44"

echo "Pulling checkpoint tarball from TAMIA..."
rsync -avP "${TAMIA_USER}@tamia.ecpia.ca:${TARBALL_REMOTE}" "$TARBALL_LOCAL"

echo "Verifying checksum..."
actual_md5=$(md5sum "$TARBALL_LOCAL" | awk '{print $1}')
if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
    echo "ERROR: checksum mismatch (got $actual_md5, expected $EXPECTED_MD5). Aborting."
    exit 1
fi
echo "Checksum OK."

echo "Extracting..."
mkdir -p trained_statedicts
tar -xf "$TARBALL_LOCAL" -C trained_statedicts

echo "Done. Checkpoints are in ./trained_statedicts/"
echo "See rorqual_ready_combos.csv for the 40 (backbone,dataset) combos to evaluate."
echo "Project name convention: full_fine_tuning_{backbone with . and / -> _}_TRADES_beta6_hpo"
echo "(same value for both --project_name and --hpo_source_project)"
