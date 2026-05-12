#!/bin/bash
# Multi-GPU parallel runner for DROID processing pipeline
# Usage:
#   bash run_parallel.sh                              # Stage 1, all episodes
#   bash run_parallel.sh --stage 2                    # Stage 2, all episodes
#   bash run_parallel.sh --limit 32                   # Limit to 32 episodes

# ---------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------
STAGE="1"
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage|-s)
            STAGE="$2"
            shift 2
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$STAGE" != "1" && "$STAGE" != "2" ]]; then
    echo "❌ Invalid stage: $STAGE (must be 1 or 2)"
    exit 1
fi

echo "🎯 Pipeline Stage: $STAGE"

# ---------------------------------------------------------
# 2. Detect available GPUs
# ---------------------------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected by nvidia-smi"
    exit 1
fi
echo "🔍 Detected $NUM_GPUS GPU(s)"

# ---------------------------------------------------------
# 3. Select script based on stage
# ---------------------------------------------------------
if [[ "$STAGE" == "1" ]]; then
    SCRIPT="compute_depth.py"
else
    SCRIPT="compute_extrinsics.py"
fi

# ---------------------------------------------------------
# 4. Full-load parallel execution
# ---------------------------------------------------------
LOGFILE="parallel_stage${STAGE}_status.log"
EXTRA_ARGS=""
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS="--limit $LIMIT"
fi

echo "🚀 Running $SCRIPT | Slots: ${NUM_GPUS}x GPU (PointWorld Static Sharding Mode) | Limit: ${LIMIT:-All}"
seq 0 $((NUM_GPUS-1)) | parallel -j "$NUM_GPUS" --ungroup --progress --joblog "$LOGFILE" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $NUM_GPUS $EXTRA_ARGS"
