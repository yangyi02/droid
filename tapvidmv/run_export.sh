#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORKERS=${1:-8}
shift || true

mkdir -p tapvidmv/logs
echo "export.py | $WORKERS workers | per-rank output: tapvidmv/logs/export.rank<N>.log"

seq 0 $((WORKERS - 1)) | parallel -j "$WORKERS" \
    "python -u tapvidmv/export.py --rank {} --world_size $WORKERS $* > tapvidmv/logs/export.rank{}.log 2>&1"

RELEASE=$(echo " $* " | sed -n 's/.* --output_root[ =]\([^ ]*\) .*/\1/p')
RELEASE=${RELEASE:-tapvidmv/data/release}
(cd "$RELEASE" && find . -name "*.partial" -prune -o -name "*.npy" -print | sed 's|^\./||' | LC_ALL=C sort > droid_file_list.txt)
echo "$(wc -l < "$RELEASE/droid_file_list.txt") files listed in $RELEASE/droid_file_list.txt"
