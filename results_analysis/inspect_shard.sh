#!/usr/bin/env bash
# inspect_shard.sh — Inspect one WebDataset shard from RobustGenBench.
# Works on Linux and macOS.
#
# Usage:
#   bash inspect_shard.sh
#
# Paste the FULL output back to Claude.

set -euo pipefail

URL="https://huggingface.co/datasets/legolasflagstaff/RobustGenBench/resolve/main/uc-merced-land-use-dataset_processed.tar.zst"
WORKDIR="$(mktemp -d)"
ARCHIVE="$WORKDIR/uc-merced.tar.zst"
TARFILE="$WORKDIR/uc-merced.tar"

if   command -v sha256sum >/dev/null 2>&1; then SHA256="sha256sum"
elif command -v shasum    >/dev/null 2>&1; then SHA256="shasum -a 256"
elif command -v openssl   >/dev/null 2>&1; then SHA256="openssl dgst -sha256"
else                                            SHA256=""
fi

echo "=========================================================="
echo "STEP 1 — environment"
echo "=========================================================="
uname -a
for tool in curl zstd tar file head awk sort uniq; do
  command -v "$tool" >/dev/null 2>&1 && echo "[ok]   $tool"     || echo "[MISS] $tool"
done
echo "sha256 tool: ${SHA256:-NONE FOUND}"
echo

echo "=========================================================="
echo "STEP 2 — download (uc-merced clean shard, ~218 MB)"
echo "=========================================================="
if [ ! -s "$ARCHIVE" ]; then
  curl -sSL --fail -o "$ARCHIVE" "$URL"
fi
ls -l "$ARCHIVE"
if [ -n "$SHA256" ]; then
  echo "sha256:"
  $SHA256 "$ARCHIVE"
fi
echo

echo "=========================================================="
echo "STEP 3 — decompress"
echo "=========================================================="
zstd -d --keep -f "$ARCHIVE" -o "$TARFILE"
ls -l "$TARFILE"
echo

echo "=========================================================="
echo "STEP 4 — first 30 entries of the tar"
echo "=========================================================="
tar -tf "$TARFILE" | head -30
echo

echo "=========================================================="
echo "STEP 5 — total entries + extension distribution"
echo "=========================================================="
TOTAL="$(tar -tf "$TARFILE" | wc -l | tr -d ' ')"
echo "total entries: $TOTAL"
echo "extensions (top 20):"
tar -tf "$TARFILE" \
  | awk -F. '{ if (NF>=2) print "."$NF; else print "(no extension)" }' \
  | sort | uniq -c | sort -rn | head -20
echo

echo "=========================================================="
echo "STEP 6 — distinct sample basenames (first 5)"
echo "=========================================================="
tar -tf "$TARFILE" \
  | awk -F/ '{print $NF}' \
  | sed -E 's/\.[^.]+$//' \
  | sort -u | head -5
echo

echo "=========================================================="
echo "STEP 7 — peek at one full sample"
echo "=========================================================="
mkdir -p "$WORKDIR/extract"

FIRST_BASENAME="$(tar -tf "$TARFILE" \
  | awk -F/ '{print $NF}' \
  | sed -E 's/\.[^.]+$//' \
  | head -1)"
echo "first basename: $FIRST_BASENAME"
echo

MEMBERS="$(tar -tf "$TARFILE" | grep -E "(^|/)${FIRST_BASENAME}\." | head -10 || true)"
if [ -n "$MEMBERS" ]; then
  # shellcheck disable=SC2086
  ( cd "$WORKDIR/extract" && tar -xf "$TARFILE" $MEMBERS ) 2>/dev/null || true
fi

echo "files extracted for that sample:"
ls -l "$WORKDIR/extract"
echo
echo "MIME / type per file:"
find "$WORKDIR/extract" -type f -exec file {} \;
echo

for f in "$WORKDIR/extract"/*; do
  [ -f "$f" ] || continue
  case "$f" in
    *.jpg|*.jpeg|*.png|*.webp|*.JPG|*.JPEG|*.PNG|*.WEBP) ;;
    *)
      echo "----- ${f##*/} (first 500 bytes, text) -----"
      head -c 500 "$f"
      echo
      echo "(end of file preview)"
      echo
      ;;
  esac
done

echo "=========================================================="
echo "DONE. Paste the entire output of this script back to Claude."
echo "=========================================================="
echo "(Workdir was $WORKDIR — safe to delete.)"