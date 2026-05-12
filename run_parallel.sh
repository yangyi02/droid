#!/bin/bash
# Multi-GPU parallel runner for DROID processing pipeline
# Usage:
#   bash run_parallel.sh                              # Stage 1, all episodes
#   bash run_parallel.sh --stage 2                    # Stage 2, all episodes
#   bash run_parallel.sh --stage 2 --file eps.txt     # Stage 2, custom list

# ---------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------
CUSTOM_FILE=""
STAGE="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file|-f)
            CUSTOM_FILE="$2"
            shift 2
            ;;
        --stage|-s)
            STAGE="$2"
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
# 2. Episode list selection
# ---------------------------------------------------------
if [ -n "$CUSTOM_FILE" ]; then
    if [ ! -f "$CUSTOM_FILE" ]; then
        echo "❌ File not found: $CUSTOM_FILE"
        exit 1
    fi
    TARGET_LIST="$CUSTOM_FILE"
    echo "📂 Using custom episode list: $TARGET_LIST ($(wc -l < "$TARGET_LIST") episodes)"
else
    if [[ "$STAGE" == "1" ]]; then
        DEFAULT_LIST="episodes.txt"
        if [ ! -f "$DEFAULT_LIST" ]; then
            echo "Generating full episode list $DEFAULT_LIST..."
            python -c "from process_droid_stage1 import load_metadata; _, _, _, _, valid_list = load_metadata(); open('$DEFAULT_LIST', 'w').write('\n'.join(valid_list))"
        fi
    else
        # Stage 2: default to episodes completed by stage 1
        DEFAULT_LIST="episodes_stage1.txt"
        if [ ! -f "$DEFAULT_LIST" ]; then
            echo "❌ $DEFAULT_LIST not found. Run stage 1 first or specify --file."
            exit 1
        fi
    fi
    TARGET_LIST="$DEFAULT_LIST"
fi

echo "📋 Episode list: $TARGET_LIST ($(wc -l < "$TARGET_LIST") episodes)"

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
# 4. Select script based on stage
# ---------------------------------------------------------
if [[ "$STAGE" == "1" ]]; then
    SCRIPT="process_droid_stage1.py"
else
    SCRIPT="process_droid_stage2.py"
fi

# ---------------------------------------------------------
# 5. Full-load parallel execution
# ---------------------------------------------------------
LOGFILE="parallel_stage${STAGE}_status.log"
echo "🚀 Running $SCRIPT | Slots: ${NUM_GPUS}x GPU (PointWorld Static Sharding Mode)"
seq 0 $((NUM_GPUS-1)) | parallel -j "$NUM_GPUS" --ungroup --progress --joblog "$LOGFILE" \
    "CUDA_VISIBLE_DEVICES={} python $SCRIPT --rank {} --world_size $NUM_GPUS"
