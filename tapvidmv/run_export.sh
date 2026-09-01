#!/bin/bash
# Parallel TAPVid-MV export. Separate from the repo-root run_parallel.sh
# because this produces the release, not a pipeline stage.
# Usage:
#   bash tapvidmv/run_export.sh                # all episodes
#   bash tapvidmv/run_export.sh --limit 32     # first 32 episodes only

cd "$(dirname "${BASH_SOURCE[0]}")"

LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--limit) LIMIT="$2"; shift 2 ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# CPU-only work: 75% of the cores, leaving headroom for IO.
JOBS=$(( $(nproc) * 3 / 4 ))

mkdir -p ../logs
echo "🚀 export_tapvidmv.py | $JOBS worker(s) | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog ../logs/parallel_export_tapvidmv_status.log \
    "python export_tapvidmv.py --rank {} --world_size $JOBS ${LIMIT:+--limit $LIMIT}"
