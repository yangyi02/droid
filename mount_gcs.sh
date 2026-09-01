#!/bin/bash
# Mount GCS buckets for DROID pipeline. Usage: bash mount_gcs.sh
# Mount points live under the repo's droid_data/, matching core.io.DATA_ROOT.
cd "$(dirname "${BASH_SOURCE[0]}")"

sudo modprobe fuse

DATA_ROOT="$(pwd)/droid_data"

fusermount -uz "$DATA_ROOT/input/robotics/droid_raw" 2>/dev/null
mkdir -p "$DATA_ROOT/input/robotics/droid_raw"
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch "$DATA_ROOT/input/robotics/droid_raw"

fusermount -uz "$DATA_ROOT/output/mv-tap" 2>/dev/null
mkdir -p "$DATA_ROOT/output/mv-tap"
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet "$DATA_ROOT/output/mv-tap"
