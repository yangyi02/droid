#!/bin/bash
# Multi-GPU parallel runner for DROID processing pipeline
# Usage:
#   bash run_parallel.sh                              # Compute depth, all episodes
#   bash run_parallel.sh --mode extrinsics            # Compute extrinsics, all episodes
#   bash run_parallel.sh --mode tracks                # Compute tracks, all episodes
#   bash run_parallel.sh --mode ablation --configs E0,E4  # Ablation, 16 GPUs
#   bash run_parallel.sh --limit 32                   # Limit to 32 episodes

# ---------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------
MODE="depth"
LIMIT=""
CONFIGS="E0,E4"
EPISODES="10"
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
        --configs|-c)
            CONFIGS="$2"
            shift 2
            ;;
        --episodes|-e)
            EPISODES="$2"
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

if [[ "$MODE" != "depth" && "$MODE" != "extrinsics" && "$MODE" != "tracks" && "$MODE" != "tracks2" && "$MODE" != "ablation" ]]; then
    echo "❌ Invalid mode: $MODE (must be 'depth', 'extrinsics', 'tracks', 'tracks2', or 'ablation')"
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
elif [[ "$MODE" == "ablation" ]]; then
    SCRIPT="run_extrinsics_ablation.py"
    OP_NAME="ablation"
elif [[ "$MODE" == "tracks2" ]]; then
    SCRIPT="compute_tracks2.py"
    OP_NAME="compute_tracks2"
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
elif [[ "$MODE" == "tracks2" ]]; then
    # tracks2 is CPU-only, use 75% of available CPU cores (leaving headroom for IO)
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

if [[ "$MODE" == "ablation" ]]; then
    # Ablation mode: pass --configs and --episodes, no --limit
    EXTRA_ARGS="--configs $CONFIGS --episodes $EPISODES"
fi

echo "🚀 Running $SCRIPT | Slots: ${PARALLEL_JOBS}x Worker(s) | Limit: ${LIMIT:-All}"
seq 0 $((PARALLEL_JOBS-1)) | parallel -j "$PARALLEL_JOBS" --ungroup --progress --joblog "$LOGFILE" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $PARALLEL_JOBS $EXTRA_ARGS"

# Post-processing: aggregate ablation results
if [[ "$MODE" == "ablation" ]]; then
    echo ""
    echo "📊 Aggregating ablation results..."
    python run_extrinsics_ablation.py --configs "$CONFIGS" --episodes "$EPISODES" --summarize
fi
