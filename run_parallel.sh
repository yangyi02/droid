#!/bin/bash
# Multi-GPU parallel runner for the DROID pipeline stages.
# Usage:
#   bash run_parallel.sh --mode depth               # all episodes
#   bash run_parallel.sh --mode tracks --limit 32   # first 32 episodes only

cd "$(dirname "${BASH_SOURCE[0]}")"

declare -A SCRIPTS=(
    [depth]=compute_depth.py
    [extrinsics]=compute_extrinsics.py
    [tracks]=compute_tracks.py
    [metrics]=compute_metrics.py
)

MODE=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)  MODE="$2";  shift 2 ;;
        -l|--limit) LIMIT="$2"; shift 2 ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Only look up a mode that was actually given: an empty associative-array
# subscript is an error in bash, not an empty result.
SCRIPT=${MODE:+${SCRIPTS[$MODE]}}
[ -n "$SCRIPT" ] || {
    echo "❌ Usage: bash run_parallel.sh --mode <${!SCRIPTS[*]}> [--limit N]" >&2
    exit 1
}

# One worker per GPU, except CPU-only tracks, which takes 75% of the cores
# so there is headroom for IO.
if [[ "$MODE" == tracks ]]; then
    JOBS=$(( $(nproc) * 3 / 4 ))
else
    JOBS=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$JOBS" -gt 0 ] || { echo "❌ No GPUs detected by nvidia-smi" >&2; exit 1; }
fi

echo "🚀 $SCRIPT | $JOBS worker(s) | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog "parallel_$(basename "$SCRIPT" .py)_status.log" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $JOBS ${LIMIT:+--limit $LIMIT}"
