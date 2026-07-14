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
# 1. Git Submodules (s2m2, co-tracker, PointWorld)
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
    opencv-python \
    scipy \
    tqdm \
    h5py \
    mediapy \
    yourdfpy \
    plotly \
    pyrender

# SAM is pip-installable directly from GitHub
pip install -q git+https://github.com/facebookresearch/segment-anything.git

# co-tracker deps (vggt is installed on-demand in compute_extrinsics.py)
pip install -q "$THIRD_PARTY/co-tracker"

# TAPNext++ deps (optional, for compute_2d_tracks.py --method tapnext)
pip install -q git+https://github.com/google-deepmind/tapnet.git 2>/dev/null || true
pip install -q git+https://github.com/google-deepmind/recurrentgemma.git@main 2>/dev/null || true

echo "✅ Python dependencies installed."

# ---------------------------------------------------------
# 3. Model Weights
# ---------------------------------------------------------
echo ""
echo "⬇️  [3/4] Downloading model weights..."

# Install huggingface-cli for reliable HF downloads (xet CDN rejects wget/curl)
pip install -q "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1  # fast Rust-based transfer

# Helper: download from HuggingFace via huggingface-cli (the only reliable method)
# Usage: hf_download <repo_id> <filename> <dest_dir>
hf_download() {
    local repo="$1"
    local filename="$2"
    local dest_dir="$3"
    local dest="$dest_dir/$filename"
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -gt 1000 ]; then
        echo "  ⏭️  $filename already exists ($(du -h "$dest" | cut -f1)), skipping."
        return 0
    fi
    rm -f "$dest"  # remove any corrupt/incomplete file
    echo "  ⬇️  Downloading $filename from $repo..."
    huggingface-cli download "$repo" "$filename" --local-dir "$dest_dir"
    echo "  ✅ $filename downloaded ($(du -h "$dest" | cut -f1))."
}

# S2M2 weights
S2M2_WEIGHTS="$THIRD_PARTY/s2m2/weights/pretrain_weights"
mkdir -p "$S2M2_WEIGHTS"
hf_download "minimok/s2m2" "CH384NTR3.pth" "$S2M2_WEIGHTS"

# CoTracker weights
COTRACKER_WEIGHTS="$THIRD_PARTY/co-tracker/weights"
mkdir -p "$COTRACKER_WEIGHTS"
hf_download "facebook/cotracker3" "scaled_offline.pth" "$COTRACKER_WEIGHTS"
# Rename to match expected filename
if [ -f "$COTRACKER_WEIGHTS/scaled_offline.pth" ] && [ ! -f "$COTRACKER_WEIGHTS/cotracker3_offline.pth" ]; then
    cp "$COTRACKER_WEIGHTS/scaled_offline.pth" "$COTRACKER_WEIGHTS/cotracker3_offline.pth"
fi

# TAPNext++ weights (512×512 model) — from Google Cloud Storage, wget works fine
TAPNEXT_WEIGHTS="$THIRD_PARTY/tapnext_weights"
mkdir -p "$TAPNEXT_WEIGHTS"
TAPNEXT_PTH="$TAPNEXT_WEIGHTS/tapnextpp_512.ckpt"
if [ ! -f "$TAPNEXT_PTH" ]; then
    echo "  ⬇️  Downloading TAPNext++ 512 checkpoint..."
    wget --progress=bar:force -O "$TAPNEXT_PTH" \
        "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
    echo "  ✅ TAPNext++ weights downloaded."
else
    echo "  ⏭️  TAPNext++ weights already exist, skipping."
fi

# SAM weights — from Meta CDN, wget works fine
SAM_WEIGHTS="$THIRD_PARTY/sam_weights"
mkdir -p "$SAM_WEIGHTS"
SAM_PTH="$SAM_WEIGHTS/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_PTH" ]; then
    echo "  ⬇️  Downloading SAM ViT-H weights..."
    wget --progress=bar:force -O "$SAM_PTH" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "  ✅ SAM weights downloaded."
else
    echo "  ⏭️  SAM weights already exist, skipping."
fi

# VGGT: both package and weights are installed/downloaded on-demand in compute_extrinsics.py
echo "  ℹ️  VGGT: installed on-demand from GitHub + weights from HuggingFace Hub at first use."

echo "✅ All model weights ready."

# ---------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------
echo ""
echo "================================================"
echo "🎉 Setup complete! Directory structure:"
echo ""
echo "  droid/"
echo "  ├── compute_depth.py       (Stage 1: stereo depth)"
echo "  ├── compute_extrinsics.py  (Stage 2: camera calibration)"
echo "  ├── compute_tracks.py      (Stage 3: 3D point tracking)"
echo "  ├── core/                  (shared modules)"
echo "  │   ├── geometry.py, io.py, depth.py, physics.py, tracking.py"
echo "  ├── utils/visualization.py"
echo "  ├── pipeline.ipynb         (Colab notebook)"
echo "  ├── run_parallel.sh"
echo "  └── third_party/           (submodules + weights)"
echo ""
echo "Run pipeline:"
echo "  bash run_parallel.sh                    # Stage 1: depth"
echo "  bash run_parallel.sh --mode extrinsics  # Stage 2: extrinsics"
echo "  bash run_parallel.sh --mode tracks      # Stage 3: tracks"
echo "================================================"
