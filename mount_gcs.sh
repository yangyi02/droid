#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_ROOT="$(pwd)/data"
INPUT_DIR="$DATA_ROOT/input/robotics/droid_raw"
OUTPUT_DIR="$DATA_ROOT/output/mv-tap"

fusermount -uz "$INPUT_DIR" 2>/dev/null
mkdir -p "$INPUT_DIR"
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch "$INPUT_DIR"

fusermount -uz "$OUTPUT_DIR" 2>/dev/null
mkdir -p "$OUTPUT_DIR"
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet "$OUTPUT_DIR"
