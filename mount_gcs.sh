#!/bin/bash
# Mount GCS buckets for DROID pipeline. Usage: bash mount_gcs.sh
# Mount points live under the repo's droid_data/, matching core.io.DATA_ROOT.
cd "$(dirname "${BASH_SOURCE[0]}")"

sudo modprobe fuse

DATA_ROOT="$(pwd)/droid_data"
INPUT_DIR="$DATA_ROOT/input/robotics/droid_raw"
OUTPUT_DIR="$DATA_ROOT/output/mv-tap"

fusermount -uz "$INPUT_DIR" 2>/dev/null
mkdir -p "$INPUT_DIR"
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch "$INPUT_DIR"

fusermount -uz "$OUTPUT_DIR" 2>/dev/null
mkdir -p "$OUTPUT_DIR"
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet "$OUTPUT_DIR"

# gcsfuse can report success without leaving a usable mount, and the pipeline
# would then just read empty directories. Say so here instead.
mountpoint -q "$INPUT_DIR" && mountpoint -q "$OUTPUT_DIR" \
    || { echo "❌ Mount failed — the pipeline will not find its data." >&2; exit 1; }
echo "🎉 Mounted under $DATA_ROOT"
