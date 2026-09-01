#!/bin/bash
# Multi-GPU parallel runner for the DROID pipeline stages.
# Usage:
#   bash run_parallel.sh                    # depth (the default), all episodes
#   bash run_parallel.sh --mode tracks      # any stage in SCRIPTS below
#   bash run_parallel.sh --limit 32         # first 32 episodes only
#   bash run_parallel.sh --jobs 4           # override the worker count

cd "$(dirname "${BASH_SOURCE[0]}")"

declare -A SCRIPTS=(
    [depth]=compute_depth.py
    [extrinsics]=compute_extrinsics.py
    [tracks]=compute_tracks.py
    [metrics]=compute_metrics.py
    [export]=tapvidmv/export_tapvid3d.py
)

MODE=depth
LIMIT=""
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)  MODE="$2";  shift 2 ;;
        -l|--limit) LIMIT="$2"; shift 2 ;;
        -j|--jobs)  JOBS="$2";  shift 2 ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT="${SCRIPTS[$MODE]}"
[ -n "$SCRIPT" ] || { echo "❌ Invalid mode: $MODE (one of: ${!SCRIPTS[*]})" >&2; exit 1; }

# One worker per GPU, except the CPU-only stages, which take 75% of the cores
# so there is headroom for IO.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc 2>/dev/null || echo 16)
if [ -z "$JOBS" ]; then
    if [[ "$MODE" == tracks || "$MODE" == export ]]; then
        JOBS=$(( NUM_CPUS * 3 / 4 ))
        [ "$JOBS" -lt 1 ] && JOBS=1
    elif [ "$NUM_GPUS" -gt 0 ]; then
        JOBS=$NUM_GPUS
    else
        echo "❌ No GPUs detected by nvidia-smi" >&2
        exit 1
    fi
fi

echo "🚀 $SCRIPT | $JOBS worker(s) ($NUM_GPUS GPU, $NUM_CPUS CPU) | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog "parallel_$(basename "$SCRIPT" .py)_status.log" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $JOBS ${LIMIT:+--limit $LIMIT}"
