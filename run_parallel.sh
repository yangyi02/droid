#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STAGE=${1:?usage: run_parallel.sh <depth|extrinsics|tracks> [--config.x=y ...]}
shift
GPUS=$(nvidia-smi -L | wc -l)

mkdir -p logs
echo "compute_$STAGE.py | $GPUS GPU(s)"
echo "per-rank output: logs/$STAGE.rank<N>.log | exit codes: logs/$STAGE.log"

seq 0 $((GPUS - 1)) | parallel -j "$GPUS" --progress --joblog "logs/$STAGE.log" \
    "CUDA_VISIBLE_DEVICES={} python -u compute_$STAGE.py \
        --config.runner.rank {} \
        --config.runner.world_size $GPUS \
        $* > logs/$STAGE.rank{}.log 2>&1"
