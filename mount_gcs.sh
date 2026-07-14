#!/bin/bash
# Mount GCS buckets for DROID pipeline

umount ~/droid_data/input/robotics/droid_raw
mkdir -p ~/droid_data/input/robotics/droid_raw
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch ~/droid_data/input/robotics/droid_raw

umount ~/droid_data/output/mv-tap
mkdir -p ~/droid_data/output/mv-tap
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet ~/droid_data/output/mv-tap
