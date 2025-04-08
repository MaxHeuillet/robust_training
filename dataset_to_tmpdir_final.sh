#!/bin/bash

DATA="$1"

echo "Processing dataset: ${DATA}"

# Create a directory for the dataset
DEST_DIR="$SLURM_TMPDIR/data/${DATA}"
mkdir -p "$DEST_DIR"

ARCHIVE_PATH=~/scratch/data/${DATA}_processed.tar.zst

extract_tar_zst() {
    echo "Extracting ${ARCHIVE_PATH} into ${DEST_DIR} with tar + zstd..."
    tar -I zstd -xf "$ARCHIVE_PATH" -C "$DEST_DIR"
}

extract_tar_zst

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA} archive." >&2
    exit 1
fi

echo "✅ Extraction of ${DATA} completed successfully at $(date)."
