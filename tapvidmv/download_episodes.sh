#!/bin/bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

URL="https://storage.googleapis.com/dm-tapnet/mv-tap/droid/tapvidmv"
DEST=~/Desktop/droid
FILES="tracks_xyz.npy queries_xytv.npy"
VIEW_FILES="depth.npy extrinsics_w2c.npy foreground_mask.npy images_jpeg_bytes.npy intrinsics.npy visibility.npy"
JOBS=10

tasks=()
while read ep; do
  enc=$(echo "$ep" | sed 's/+/%2B/g')
  mkdir -p "$DEST/$ep"/{0,1,2}

  for f in $FILES; do
    [ -f "$DEST/$ep/$f" ] && continue
    tasks+=("$DEST/$ep/$f $URL/$enc/$f")
  done
  for v in 0 1 2; do
    for f in $VIEW_FILES; do
      [ -f "$DEST/$ep/$v/$f" ] && continue
      tasks+=("$DEST/$ep/$v/$f $URL/$enc/$v/$f")
    done
  done
done < episodes_eval50.txt

echo "Total: ${#tasks[@]} files, $JOBS parallel"
printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -L1 bash -c \
  'wget -q -O "$1" "$2" || rm -f "$1"' _

echo "Done!"
