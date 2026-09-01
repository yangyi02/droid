#!/bin/bash
# Parallel TAPVid-MV export. Separate from the repo-root run_parallel.sh
# because this produces the release, not a pipeline stage.
# Usage:
#   bash tapvidmv/run_export.sh                 # all episodes
#   bash tapvidmv/run_export.sh --limit 32      # first 32 episodes only
#   bash tapvidmv/run_export.sh --jobs 8        # override the worker count

cd "$(dirname "${BASH_SOURCE[0]}")"

LIMIT=""
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--limit) LIMIT="$2"; shift 2 ;;
        -j|--jobs)  JOBS="$2";  shift 2 ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# CPU-only work: 75% of the cores, leaving headroom for IO.
NUM_CPUS=$(nproc 2>/dev/null || echo 16)
if [ -z "$JOBS" ]; then
    JOBS=$(( NUM_CPUS * 3 / 4 ))
    [ "$JOBS" -lt 1 ] && JOBS=1
fi

echo "🚀 export_tapvidmv.py | $JOBS worker(s) ($NUM_CPUS CPU) | limit: ${LIMIT:-all}"
seq 0 $((JOBS - 1)) | parallel -j "$JOBS" --ungroup --progress \
    --joblog parallel_export_tapvidmv_status.log \
    "python export_tapvidmv.py --rank {} --world_size $JOBS ${LIMIT:+--limit $LIMIT}"
