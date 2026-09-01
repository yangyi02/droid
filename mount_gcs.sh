#!/bin/bash
# Mount GCS buckets for DROID pipeline. Usage: bash mount_gcs.sh
# Mount points live under the repo's droid_data/, matching core.io.DATA_ROOT.
# Find the repo root by walking up from this script to the directory holding
# core/io.py — the very file core.io.DATA_ROOT anchors on, so the shell and
# Python can never disagree about where droid_data/ is. Anchoring on a marker
# instead of a fixed "../" also means this keeps working if the script is moved
# into a subdirectory. (Inlined rather than shared: bootstrap code that locates
# the repo cannot itself live inside the repo it has yet to locate.)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -f "$REPO_ROOT/core/io.py" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -f "$REPO_ROOT/core/io.py" ]; then
    echo "❌ Not inside the droid repo: no core/io.py found above ${BASH_SOURCE[0]}" >&2
    exit 1
fi
cd "$REPO_ROOT"

sudo modprobe fuse

DATA_ROOT="$REPO_ROOT/droid_data"
INPUT_DIR="$DATA_ROOT/input/robotics/droid_raw"
OUTPUT_DIR="$DATA_ROOT/output/mv-tap"

echo "📁 Mounting under $DATA_ROOT"

fusermount -uz "$INPUT_DIR" 2>/dev/null
mkdir -p "$INPUT_DIR"
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch "$INPUT_DIR"

fusermount -uz "$OUTPUT_DIR" 2>/dev/null
mkdir -p "$OUTPUT_DIR"
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet "$OUTPUT_DIR"

# A wrong mount point fails silently — gcsfuse succeeds, then every stage reads
# an empty directory. Assert the buckets really are mounted where the pipeline
# will look for them before declaring success.
echo "🔍 Verifying mounts"
status=0
for d in "$INPUT_DIR" "$OUTPUT_DIR"; do
    if mountpoint -q "$d" 2>/dev/null || \
       [ "$(findmnt -rn -o TARGET --target "$d" 2>/dev/null)" = "$d" ]; then
        echo "  ✅ $d"
    else
        echo "  ❌ $d is not a mount point"
        status=1
    fi
done
[ "$status" -eq 0 ] && echo "🎉 Mounts ready" || echo "⚠️  Some mounts failed — the pipeline will not find its data."
exit "$status"
