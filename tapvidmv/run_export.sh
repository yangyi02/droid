#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORKERS=${1:-8}
shift || true

mkdir -p tapvidmv/logs
echo "export.py | $WORKERS workers | per-rank output: tapvidmv/logs/export.rank<N>.log"

seq 0 $((WORKERS - 1)) | parallel -j "$WORKERS" \
    "python -u tapvidmv/export.py --rank {} --world_size $WORKERS $* > tapvidmv/logs/export.rank{}.log 2>&1"
