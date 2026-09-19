#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EPISODE=${1:?usage: serve_review.sh <episode_id> [port]}
PORT=${2:-9876}

echo "Forward port $PORT to your laptop, then run there: rerun rerun+http://127.0.0.1:$PORT/proxy"

exec venv/bin/rerun --serve-grpc --bind 127.0.0.1 --port "$PORT" --server-memory-limit 50% \
    "data/output/droid/review/$EPISODE.rrd"
