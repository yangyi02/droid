#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_ROOT="$(pwd)/data"
INPUT_DIR="$DATA_ROOT/input/robotics/droid_raw"

fusermount -uz "$INPUT_DIR" 2>/dev/null
mkdir -p "$INPUT_DIR"
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch "$INPUT_DIR"
