#!/bin/bash
set -euo pipefail

STAGE=${1:?usage: entrypoint.sh <depth|extrinsics|scene|tracks|metrics|review> [--config.x=y ...]}
shift
cd /app

echo "== $(hostname) | stage=${STAGE} | $*"
nvidia-smi -L || true
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "GPU memory in use: ${used:-?} MiB"
  if [ -z "${used}" ] || [ "${used}" -lt 2048 ]; then break; fi
  [ "$i" -eq 60 ] && echo "GPU still busy after 10 min; exiting" && exit 1
  sleep 10
done
df -h /dev/disk-scratch /app/data/input/robotics/droid_raw /app/data/output/droid 2>/dev/null || true

exec python -u xm/run_stage.py "${STAGE}" "$@"
