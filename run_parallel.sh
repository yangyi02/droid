#!/bin/bash
# One pipeline stage, one worker per GPU, episodes sharded by rank.
#
#   bash run_parallel.sh tracks      # every episode
#   bash run_parallel.sh depth 32    # first 32 episodes
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STAGE=${1:?usage: run_parallel.sh <depth|extrinsics|tracks|metrics> [limit]}
LIMIT=${2:-}
GPUS=$(nvidia-smi -L | wc -l)

mkdir -p logs
echo "compute_$STAGE.py | $GPUS GPU(s) | ${LIMIT:-all} episodes"

seq 0 $((GPUS - 1)) | parallel -j "$GPUS" --ungroup --progress --joblog "logs/$STAGE.log" \
    "CUDA_VISIBLE_DEVICES={} python compute_$STAGE.py \
        --config.runner.rank {} \
        --config.runner.world_size $GPUS \
        ${LIMIT:+--config.runner.limit $LIMIT}"
