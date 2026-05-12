#!/bin/bash
# One-click GCS bucket mount script for datasets

# Unmount a mount point if it is already mounted (prevents fusermount errors).
safe_mount() {
  local mount_point="$1"
  shift
  # "$@" = remaining args passed to gcsfuse

  if mountpoint -q "$mount_point" 2>/dev/null; then
    echo "⚠️  $mount_point is already mounted — unmounting first..."
    fusermount -u "$mount_point" || umount "$mount_point" 2>/dev/null
  fi

  mkdir -p "$mount_point"
  gcsfuse "$@" "$mount_point"
}

echo "========================================="
echo "1. Mounting DROID raw video & trajectory dataset"
echo "========================================="
if safe_mount ~/droid_data/input/robotics/droid_raw \
     --implicit-dirs --only-dir robotics/droid_raw gresearch; then
  echo "✅ Input dataset mounted successfully."
else
  echo "❌ Failed to mount input dataset." >&2
  exit 1
fi

echo "========================================="
echo "2. Mounting output / multi-view depth storage bucket"
echo "========================================="
if safe_mount ~/droid_data/output/mv-tap \
     --implicit-dirs --only-dir mv-tap dm-tapnet; then
  echo "✅ Output bucket mounted successfully."
else
  echo "❌ Failed to mount output bucket." >&2
  exit 1
fi

echo "🚀 All mounts ready — pipeline is good to go!"
