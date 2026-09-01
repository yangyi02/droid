#!/bin/bash
# Multi-GPU parallel runner for DROID processing pipeline
# Usage:
#   bash run_parallel.sh                              # Compute depth, all episodes
#   bash run_parallel.sh --mode extrinsics            # Compute extrinsics, all episodes
#   bash run_parallel.sh --mode tracks                # Compute tracks, all episodes
#   bash run_parallel.sh --mode export                # Export to TAPVid-3D format, all episodes
#   bash run_parallel.sh --mode metrics               # Evaluate quality metrics, all episodes
#   bash run_parallel.sh --limit 32                   # Limit to 32 episodes

# Find the repo root by walking up from this script to the directory holding
# core/io.py — the very file core.io.DATA_ROOT anchors on, so the shell and
# Python can never disagree about where droid_data/ is. Anchoring on a marker
# instead of a fixed "../" also means this keeps working if the script is moved
# into a subdirectory. (Inlined rather than shared: bootstrap code that locates
# the repo cannot itself live inside the repo it has yet to locate.)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -f "$REPO_ROOT/core/io.py" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -f "$REPO_ROOT/core/io.py" ]; then
    echo "❌ Not inside the droid repo: no core/io.py found above ${BASH_SOURCE[0]}" >&2
    exit 1
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------
MODE="depth"
LIMIT=""
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "depth" && "$MODE" != "extrinsics" && "$MODE" != "tracks" && "$MODE" != "metrics" && "$MODE" != "export" ]]; then
    echo "❌ Invalid mode: $MODE (must be 'depth', 'extrinsics', 'tracks', 'metrics', or 'export')"
    exit 1
fi

# ---------------------------------------------------------
# 2. Select script based on mode
# ---------------------------------------------------------
if [[ "$MODE" == "depth" ]]; then
    SCRIPT="compute_depth.py"
    OP_NAME="compute_depth"
elif [[ "$MODE" == "extrinsics" ]]; then
    SCRIPT="compute_extrinsics.py"
    OP_NAME="compute_extrinsics"
elif [[ "$MODE" == "metrics" ]]; then
    SCRIPT="evaluate_episodes.py"
    OP_NAME="evaluate_metrics"
elif [[ "$MODE" == "export" ]]; then
    SCRIPT="tapvidmv/export_tapvid3d.py"
    OP_NAME="export_tapvid3d"
else
    SCRIPT="compute_tracks.py"
    OP_NAME="compute_tracks"
fi

echo "🎯 Running: $SCRIPT ($OP_NAME)"

# ---------------------------------------------------------
# 3. Detect available GPUs
# ---------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc 2>/dev/null || echo 16)

if [ -n "$JOBS" ]; then
    PARALLEL_JOBS="$JOBS"
elif [[ "$MODE" == "tracks" || "$MODE" == "export" ]]; then
    # tracks and export are CPU-only, use 75% of available CPU cores (leaving headroom for IO)
    PARALLEL_JOBS=$(( NUM_CPUS * 3 / 4 ))
    if [ "$PARALLEL_JOBS" -lt 1 ]; then PARALLEL_JOBS=1; fi
else
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "❌ No GPUs detected by nvidia-smi"
        exit 1
    fi
    PARALLEL_JOBS="$NUM_GPUS"
fi
echo "🔍 Detected $NUM_GPUS GPU(s), $NUM_CPUS CPU core(s) -> Using $PARALLEL_JOBS parallel worker(s)"

# ---------------------------------------------------------
# 4. Full-load parallel execution
# ---------------------------------------------------------
LOGFILE="parallel_${OP_NAME}_status.log"
EXTRA_ARGS=""
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS="--limit $LIMIT"
fi

echo "🚀 Running $SCRIPT | Slots: ${PARALLEL_JOBS}x Worker(s) | Limit: ${LIMIT:-All}"
seq 0 $((PARALLEL_JOBS-1)) | parallel -j "$PARALLEL_JOBS" --ungroup --progress --joblog "$LOGFILE" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $PARALLEL_JOBS $EXTRA_ARGS"

