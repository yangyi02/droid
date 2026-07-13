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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "depth" && "$MODE" != "extrinsics" && "$MODE" != "tracks" && "$MODE" != "ablation" ]]; then
    echo "❌ Invalid mode: $MODE (must be 'depth', 'extrinsics', 'tracks', or 'ablation')"
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
else
    SCRIPT="compute_tracks.py"
    OP_NAME="compute_tracks"
fi

echo "🎯 Running: $SCRIPT ($OP_NAME)"

# ---------------------------------------------------------
# 3. Detect available GPUs
# ---------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected by nvidia-smi"
    exit 1
fi
echo "🔍 Detected $NUM_GPUS GPU(s)"

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

echo "🚀 Running $SCRIPT | Slots: ${NUM_GPUS}x GPU (PointWorld Static Sharding Mode) | Limit: ${LIMIT:-All}"
seq 0 $((NUM_GPUS-1)) | parallel -j "$NUM_GPUS" --ungroup --progress --joblog "$LOGFILE" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $NUM_GPUS $EXTRA_ARGS"

# Post-processing: aggregate ablation results
if [[ "$MODE" == "ablation" ]]; then
    echo ""
    echo "📊 Aggregating ablation results..."
    python run_extrinsics_ablation.py --configs "$CONFIGS" --episodes "$EPISODES" --summarize
fi
