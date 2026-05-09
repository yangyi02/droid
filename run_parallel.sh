#!/bin/bash
# Multi-GPU parallel runner (supports production and debug modes)

# Parse debug flag
DEBUG_MODE=0
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    DEBUG_MODE=1
    echo "🐛 Debug mode enabled..."
fi

# ---------------------------------------------------------
# 1. Episode list generation and selection
# ---------------------------------------------------------
if [ ! -f "episodes.txt" ]; then
    echo "Generating full episode list episodes.txt..."
    python -c "from process_droid_stage1 import load_metadata; _, _, _, _, valid_list = load_metadata(); open('episodes.txt', 'w').write('\n'.join(valid_list))"
fi

TARGET_LIST="episodes.txt"

if [ $DEBUG_MODE -eq 1 ]; then
    echo "Extracting first 4 entries for debug list..."
    head -n 4 episodes.txt > episodes_debug.txt
    TARGET_LIST="episodes_debug.txt"
fi

# ---------------------------------------------------------
# 2. Full-load parallel execution
# ---------------------------------------------------------
echo "🚀 Running list: $TARGET_LIST | Slots: 2x A100"
cat "$TARGET_LIST" | parallel -j 2 --ungroup --progress --joblog parallel_status.log \
    "CUDA_VISIBLE_DEVICES=\$(({%}-1)) python process_droid_stage1.py --ep_list '{}'"
