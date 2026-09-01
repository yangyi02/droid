#!/bin/bash
# Parallel TAPVid-MV export. Separate from the repo-root run_parallel.sh
# because this produces the release, not a pipeline stage.
#
# Runs over the selected episodes only. Exporting re-encodes every frame to
# JPEG and writes the depth maps, so it is by far the most expensive step —
# doing it before selection meant paying that for thousands of episodes to
# keep fifty.
#
# Usage:
#   bash tapvidmv/run_export.sh                             # episodes_eval50.txt
#   bash tapvidmv/run_export.sh --list episodes_eval150.txt # a different set
#   bash tapvidmv/run_export.sh --list all                  # everything with tracks
#   bash tapvidmv/run_export.sh --limit 8                   # first 8 of the list

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

# CPU-only work: 75% of the cores, leaving headroom for IO.
JOBS=$(( $(nproc) * 3 / 4 ))
[ "$LIST" != "all" ] && JOBS=$(( $(wc -l < "$LIST") < JOBS ? $(wc -l < "$LIST") : JOBS ))
[ "$JOBS" -lt 1 ] && JOBS=1

mkdir -p ../logs
echo "🚀 export_tapvidmv.py | $JOBS worker(s) | list: $LIST | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog ../logs/export_tapvidmv_status.log \
    "python export_tapvidmv.py --rank {} --world_size $JOBS --episode_list $LIST ${LIMIT:+--limit $LIMIT}"
