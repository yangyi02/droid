#!/bin/bash
# Mount GCS buckets for DROID pipeline
sudo modprobe fuse

fusermount -uz ~/droid_data/input/robotics/droid_raw 2>/dev/null
mkdir -p ~/droid_data/input/robotics/droid_raw
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch ~/droid_data/input/robotics/droid_raw

fusermount -uz ~/droid_data/output/mv-tap 2>/dev/null
mkdir -p ~/droid_data/output/mv-tap
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet ~/droid_data/output/mv-tap
