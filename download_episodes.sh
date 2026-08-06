#!/bin/bash
set -uo pipefail

URL="https://storage.googleapis.com/dm-tapnet/mv-tap/droid/tapvidmv"
DEST=~/Desktop/droid
FILES="tracks_xyz.npy queries_xytv.npy"
VIEW_FILES="depth.npy extrinsics_w2c.npy foreground_mask.npy images_jpeg_bytes.npy intrinsics.npy visibility.npy"

while read ep; do
  echo "=== $ep ==="
  enc=$(echo "$ep" | sed 's/+/%2B/g')

  for f in $FILES; do
    mkdir -p "$DEST/$ep"
    [ -f "$DEST/$ep/$f" ] && continue
    wget -q -O "$DEST/$ep/$f" "$URL/$enc/$f" || rm -f "$DEST/$ep/$f"
  done

  for v in 0 1 2; do
    for f in $VIEW_FILES; do
      mkdir -p "$DEST/$ep/$v"
      [ -f "$DEST/$ep/$v/$f" ] && continue
      wget -q -O "$DEST/$ep/$v/$f" "$URL/$enc/$v/$f" || rm -f "$DEST/$ep/$v/$f"
    done
  done
done < episodes_eval50.txt

echo "Done!"
