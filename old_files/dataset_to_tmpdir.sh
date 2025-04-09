#!/bin/bash

DATA="$1"  # First argument passed from main script

echo "Processing dataset: ${DATA}"

# Create a temporary data directory
mkdir -p "$SLURM_TMPDIR/data"

ARCHIVE_PATH=~/scratch/data/${DATA}

extract_tar_zst() {
    local archive="${ARCHIVE_PATH}.tar.zst"
    echo "Extracting ${archive} with tar + zstd..."
    tar -I zstd -xf "$archive" -C "$SLURM_TMPDIR/data"
}

extract_zip() {
    local archive="${ARCHIVE_PATH}.zip"
    echo "Extracting ${archive} with unzip..."
    unzip -q "$archive" -d "$SLURM_TMPDIR/data"
}

if [ "$DATA" == "stanford_cars" ] || \
   [ "$DATA" == "uc-merced-land-use-dataset" ] || \
   [ "$DATA" == "kvasir-dataset" ]; then
    echo "Dataset is $DATA. Using unzip to extract."
    extract_zip
else
    echo "Dataset is not in the zip list. Using tar.zst to extract."
    extract_tar_zst
fi

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ${DATA} archive." >&2
    exit 1
fi

echo "Extraction of ${DATA} archive completed successfully at $(date)."
