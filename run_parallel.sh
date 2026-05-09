#!/bin/bash
# Multi-GPU parallel runner
# Usage:
#   bash run_parallel.sh                       # Run all episodes
#   bash run_parallel.sh --file episodes.txt   # Run from custom file

# ---------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------
CUSTOM_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file|-f)
            CUSTOM_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

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
    if [ ! -f "episodes.txt" ]; then
        echo "Generating full episode list episodes.txt..."
        python -c "from process_droid_stage1 import load_metadata; _, _, _, _, valid_list = load_metadata(); open('episodes.txt', 'w').write('\n'.join(valid_list))"
    fi
    TARGET_LIST="episodes.txt"
fi

# ---------------------------------------------------------
# 3. Full-load parallel execution
# ---------------------------------------------------------
echo "🚀 Running list: $TARGET_LIST | Slots: 2x A100"
cat "$TARGET_LIST" | parallel -j 2 --ungroup --progress --joblog parallel_status.log \
    "CUDA_VISIBLE_DEVICES=\$(({%}-1)) python process_droid_stage1.py --ep_list '{}'"
