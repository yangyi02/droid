#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules
echo "📦 [1/4] Submodules"
git submodule update --init --recursive

# 2. Python packages
echo "🐍 [2/4] Python packages"
pip install pybullet opencv-python scipy tqdm h5py mediapy yourdfpy plotly pyrender
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install third_party/co-tracker
pip install git+https://github.com/google-deepmind/tapnet.git
pip install git+https://github.com/google-deepmind/recurrentgemma.git@main

# 3. Model weights
echo "⬇️  [3/4] Model weights"
mkdir -p third_party/s2m2/weights/pretrain_weights third_party/co-tracker/weights third_party/tapnext_weights third_party/sam_weights

wget -nc -O third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth    "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/co-tracker/weights/cotracker3_offline.pth       "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
wget -nc -O third_party/tapnext_weights/tapnextpp_512.ckpt             "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth               "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo "  ℹ️  VGGT: downloaded on first use"

# 4. Done
echo "🎉 [4/4] Setup complete"
