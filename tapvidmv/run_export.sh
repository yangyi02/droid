#!/bin/bash
# The release export, episodes sharded across workers. CPU only -- no GPU is involved, so the
# worker count comes from what the disk can absorb rather than from nvidia-smi.
#
#   bash tapvidmv/run_export.sh                 # 8 workers
#   bash tapvidmv/run_export.sh 16 --no_depth   # anything after the count goes to export.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORKERS=${1:-8}
shift || true

mkdir -p tapvidmv/logs
echo "export.py | $WORKERS workers | per-rank output: tapvidmv/logs/export.rank<N>.log"

seq 0 $((WORKERS - 1)) | parallel -j "$WORKERS" \
    "python -u tapvidmv/export.py --rank {} --world_size $WORKERS $* > tapvidmv/logs/export.rank{}.log 2>&1"
