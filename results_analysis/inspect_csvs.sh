#!/usr/bin/env bash
# inspect_csvs.sh — Look at the manifest CSVs and metadata.json in the
# RobustGenBench shard, plus a couple of sample PNGs from each split.
#
# Usage: bash inspect_csvs.sh
# Paste the FULL output back to Claude.

set -euo pipefail

URL="https://huggingface.co/datasets/legolasflagstaff/RobustGenBench/resolve/main/uc-merced-land-use-dataset_processed.tar.zst"
WORKDIR="$(mktemp -d)"
ARCHIVE="$WORKDIR/uc-merced.tar.zst"
TARFILE="$WORKDIR/uc-merced.tar"

echo "downloading and decompressing..."
curl -sSL --fail -o "$ARCHIVE" "$URL"
zstd -d --keep -f "$ARCHIVE" -o "$TARFILE" >/dev/null
echo

echo "=========================================================="
echo "ALL non-PNG entries in the archive"
echo "=========================================================="
tar -tf "$TARFILE" | grep -v '\.png$' | sort
echo

echo "=========================================================="
echo "Extract every non-PNG file"
echo "=========================================================="
mkdir -p "$WORKDIR/extract"
NONPNG="$(tar -tf "$TARFILE" | grep -v '\.png$' || true)"
( cd "$WORKDIR/extract" && tar -xf "$TARFILE" $NONPNG )
find "$WORKDIR/extract" -type f | sort
echo

for f in $(find "$WORKDIR/extract" -type f | sort); do
  echo "=========================================================="
  echo "FILE: ${f#$WORKDIR/extract/}"
  echo "=========================================================="
  echo "size: $(wc -c < "$f") bytes; lines: $(wc -l < "$f" | tr -d ' ')"
  echo "--- first 15 lines ---"
  head -15 "$f"
  echo
  echo "--- last 5 lines ---"
  tail -5 "$f"
  echo
done

echo "=========================================================="
echo "First 3 PNGs in each subdirectory (to confirm layout)"
echo "=========================================================="
for d in train val test test_common; do
  echo "[$d/]"
  tar -tf "$TARFILE" | grep "^$d/" | head -3 || echo "  (none)"
done
echo

echo "=========================================================="
echo "Per-directory PNG counts"
echo "=========================================================="
tar -tf "$TARFILE" | grep '\.png$' | awk -F/ '{print $1}' | sort | uniq -c
echo

echo "=========================================================="
echo "DONE. Paste the entire output back to Claude."
echo "=========================================================="
echo "(Workdir was $WORKDIR — safe to delete.)"