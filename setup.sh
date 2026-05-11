#!/bin/bash
# DROID Pipeline Setup Script
# Run once after cloning the repo to set up all dependencies.
#
# Usage:
#   bash setup.sh
#
# If you cloned WITHOUT --recurse-submodules, this script will init them for you.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$SCRIPT_DIR/third_party"

echo "================================================"
echo "🚀 DROID Pipeline Setup"
echo "================================================"

# ---------------------------------------------------------
# 1. Git Submodules (s2m2, vggt, co-tracker, PointWorld)
# ---------------------------------------------------------
echo ""
echo "📦 [1/4] Initializing git submodules..."
cd "$SCRIPT_DIR"
git submodule update --init --recursive
echo "✅ Submodules ready."

# ---------------------------------------------------------
# 2. Python Dependencies
# ---------------------------------------------------------
echo ""
echo "🐍 [2/4] Installing Python dependencies..."
pip install -q \
    pybullet \
    pybullet-data \
    opencv-python \
    scipy \
    tqdm \
    h5py \
    mediapy

# SAM is pip-installable directly from GitHub
pip install -q git+https://github.com/facebookresearch/segment-anything.git

# co-tracker and vggt extra deps
pip install -q "$THIRD_PARTY/co-tracker"
pip install -q "$THIRD_PARTY/vggt"

echo "✅ Python dependencies installed."

# ---------------------------------------------------------
# 3. Model Weights
# ---------------------------------------------------------
echo ""
echo "⬇️  [3/4] Downloading model weights..."

# S2M2 weights
S2M2_WEIGHTS="$THIRD_PARTY/s2m2/weights/pretrain_weights"
mkdir -p "$S2M2_WEIGHTS"
S2M2_PTH="$S2M2_WEIGHTS/CH384NTR3.pth"
if [ ! -f "$S2M2_PTH" ] || [ "$(stat -c%s "$S2M2_PTH" 2>/dev/null || echo 0)" -lt $((100 * 1024 * 1024)) ]; then
    echo "  ⬇️  Downloading S2M2 weights..."
    wget -q -O "$S2M2_PTH" "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
    echo "  ✅ S2M2 weights downloaded."
else
    echo "  ⏭️  S2M2 weights already exist, skipping."
fi

# CoTracker weights
COTRACKER_WEIGHTS="$THIRD_PARTY/co-tracker/weights"
mkdir -p "$COTRACKER_WEIGHTS"
COTRACKER_PTH="$COTRACKER_WEIGHTS/cotracker3_offline.pth"
if [ ! -f "$COTRACKER_PTH" ]; then
    echo "  ⬇️  Downloading CoTracker3 weights..."
    wget -q -O "$COTRACKER_PTH" \
        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    echo "  ✅ CoTracker3 weights downloaded."
else
    echo "  ⏭️  CoTracker3 weights already exist, skipping."
fi

# SAM weights
SAM_WEIGHTS="$THIRD_PARTY/sam_weights"
mkdir -p "$SAM_WEIGHTS"
SAM_PTH="$SAM_WEIGHTS/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_PTH" ]; then
    echo "  ⬇️  Downloading SAM ViT-H weights..."
    wget -q -O "$SAM_PTH" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "  ✅ SAM weights downloaded."
else
    echo "  ⏭️  SAM weights already exist, skipping."
fi

# VGGT is loaded from HuggingFace Hub at runtime (no manual download needed)
echo "  ℹ️  VGGT weights: auto-downloaded from HuggingFace Hub at first run."

echo "✅ All model weights ready."

# ---------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------
echo ""
echo "================================================"
echo "🎉 Setup complete! Directory structure:"
echo ""
echo "  droid/"
echo "  ├── third_party/"
echo "  │   ├── s2m2/            (submodule)"
echo "  │   │   └── weights/     (downloaded)"
echo "  │   ├── vggt/            (submodule)"
echo "  │   ├── co-tracker/      (submodule)"
echo "  │   │   └── weights/     (downloaded)"
echo "  │   ├── PointWorld/      (submodule, branch: data)"
echo "  │   └── sam_weights/     (downloaded)"
echo "  ├── process_droid_stage1.py"
echo "  ├── process_droid_stage2.py"
echo "  └── run_parallel.sh"
echo ""
echo "Run pipeline:"
echo "  bash run_parallel.sh            # Stage 1"
echo "  bash run_parallel.sh --stage 2  # Stage 2"
echo "================================================"
