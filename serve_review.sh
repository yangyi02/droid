#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EPISODE=${1:?usage: serve_review.sh <episode_id> [viewer_port] [data_port]}
VIEWER_PORT=${2:-9090}
DATA_PORT=${3:-9876}

echo "Forward ports $VIEWER_PORT and $DATA_PORT, then open http://localhost:$VIEWER_PORT?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A$DATA_PORT%2Fproxy"

exec venv/bin/rerun --serve-web --bind 127.0.0.1 --web-viewer-port "$VIEWER_PORT" --port "$DATA_PORT" \
    --server-memory-limit 50% "tapvidmv/data/review/$EPISODE.rrd"
