#!/bin/bash
# 一键双卡并发运行包装器 (支持生产与调试双模式)

# 接收调试标识
DEBUG_MODE=0
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    DEBUG_MODE=1
    echo "🐛 开启调试模式 (Debug Mode)..."
fi

# ---------------------------------------------------------
# 1. 清单生成与选取
# ---------------------------------------------------------
if [ ! -f "episodes.txt" ]; then
    echo "生成全量跑单清单 episodes.txt..."
    python -c "from process_droid import load_metadata; _, _, _, _, valid_list = load_metadata(); open('episodes.txt', 'w').write('\n'.join(valid_list))"
fi

TARGET_LIST="episodes.txt"

if [ $DEBUG_MODE -eq 1 ]; then
    echo "提取前 4 条记录生成调试清单..."
    head -n 4 episodes.txt > episodes_debug.txt
    TARGET_LIST="episodes_debug.txt"
fi

# ---------------------------------------------------------
# 2. 满负荷并发执行
# ---------------------------------------------------------
echo "🚀 当前运行清单: $TARGET_LIST | 槽位: 2 张 A100"
cat "$TARGET_LIST" | parallel -j 2 --ungroup --progress --joblog parallel_status.log \
    "CUDA_VISIBLE_DEVICES=\$(({%}-1)) python process_droid.py --ep_list '{}'"
