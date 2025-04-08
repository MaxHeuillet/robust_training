#!/bin/bash

DATA="$1"  # First argument passed from main script

echo "Processing dataset: ${DATA}"

# Create a temporary data directory
mkdir -p "$SLURM_TMPDIR/data"

ARCHIVE_PATH=~/scratch/data/${DATA}

extract_tar_zst() {
    local archive="${ARCHIVE_PATH}_processed.tar.zst"
    echo "Extracting ${archive} with tar + zstd..."
    tar -I zstd -xf "$archive" -C "$SLURM_TMPDIR/data"
}

echo "Dataset is not in the zip list. Using tar.zst to extract."
extract_tar_zst

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA} archive." >&2
    exit 1
fi

echo "Extraction of ${DATA} archive completed successfully at $(date)."
