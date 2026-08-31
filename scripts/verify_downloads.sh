#!/bin/bash
set -uo pipefail

# episodes_eval50.txt lives at the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

URL="https://storage.googleapis.com/dm-tapnet/mv-tap/droid/tapvidmv"
DEST=~/Desktop/droid
FILES="tracks_xyz.npy queries_xytv.npy"
VIEW_FILES="depth.npy extrinsics_w2c.npy foreground_mask.npy images_jpeg_bytes.npy intrinsics.npy visibility.npy"
bad=0

while read ep; do
  enc=$(echo "$ep" | sed 's/+/%2B/g')

  for f in $FILES; do
    local="$DEST/$ep/$f"
    [ -f "$local" ] || continue
    remote_size=$(curl -sI "$URL/$enc/$f" | grep -i content-length | head -1 | awk '{print $2}' | tr -d '\r')
    local_size=$(stat -c%s "$local")
    if [ "$local_size" != "$remote_size" ]; then
      echo "BAD $ep/$f (local=${local_size} remote=${remote_size})"
      rm -f "$local"
      bad=$((bad+1))
    fi
  done

  for v in 0 1 2; do
    for f in $VIEW_FILES; do
      local="$DEST/$ep/$v/$f"
      [ -f "$local" ] || continue
      remote_size=$(curl -sI "$URL/$enc/$v/$f" | grep -i content-length | head -1 | awk '{print $2}' | tr -d '\r')
      local_size=$(stat -c%s "$local")
      if [ "$local_size" != "$remote_size" ]; then
        echo "BAD $ep/$v/$f (local=${local_size} remote=${remote_size})"
        rm -f "$local"
        bad=$((bad+1))
      fi
    done
  done
done < episodes_eval50.txt

echo "Found $bad bad files (deleted). Re-run scripts/download_episodes.sh to re-download."
