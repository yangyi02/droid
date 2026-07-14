#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
TP=third_party

# 1. Submodules
echo "📦 [1/4] Submodules"
git submodule update --init --recursive

# 2. Python packages
echo "🐍 [2/4] Python packages"
pip install -q pybullet opencv-python scipy tqdm h5py mediapy yourdfpy plotly pyrender
pip install -q git+https://github.com/facebookresearch/segment-anything.git
pip install -q "$TP/co-tracker"
pip install -q git+https://github.com/google-deepmind/tapnet.git 2>/dev/null || true
pip install -q git+https://github.com/google-deepmind/recurrentgemma.git@main 2>/dev/null || true

# 3. Model weights
echo "⬇️  [3/4] Model weights"
mkdir -p $TP/s2m2/weights/pretrain_weights $TP/co-tracker/weights $TP/tapnext_weights $TP/sam_weights

download() { [ -f "$2" ] && echo "  ⏭️  $(basename $2)" || { echo "  ⬇️  $(basename $2)"; wget -q -O "$2" "$1"; } }

download "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"            "$TP/s2m2/weights/pretrain_weights/CH384NTR3.pth"
download "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" "$TP/co-tracker/weights/cotracker3_offline.pth"
download "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"      "$TP/tapnext_weights/tapnextpp_512.ckpt"
download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"        "$TP/sam_weights/sam_vit_h_4b8939.pth"

echo "  ℹ️  VGGT: downloaded on first use"

# 4. Done
echo "🎉 [4/4] Setup complete"
