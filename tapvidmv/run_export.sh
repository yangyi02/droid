#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

LIST="episodes_eval50.txt"
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--list)  LIST="$2";  shift 2 ;;
        -l|--limit) LIMIT="$2"; shift 2 ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "$LIST" != "all" ] && [ ! -f "$LIST" ]; then
    echo "❌ Episode list not found: tapvidmv/$LIST" >&2
    echo "   Run select_episodes.py first, or pass --list all." >&2
    exit 1
fi

JOBS=$(( $(nproc) * 3 / 4 ))
[ "$LIST" != "all" ] && JOBS=$(( $(wc -l < "$LIST") < JOBS ? $(wc -l < "$LIST") : JOBS ))
[ "$JOBS" -lt 1 ] && JOBS=1

mkdir -p ../logs
echo "🚀 export_tapvidmv.py | $JOBS worker(s) | list: $LIST | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog ../logs/export_tapvidmv_status.log \
    "python export_tapvidmv.py --rank {} --world_size $JOBS --episode_list $LIST ${LIMIT:+--limit $LIMIT}"
