#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules
echo "📦 [1/5] Submodules"
git submodule update --init --recursive

# 2. Python packages
echo "🐍 [2/5] Python packages"
pip install pybullet opencv-python scipy tqdm h5py mediapy yourdfpy plotly pyrender
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install third_party/co-tracker
pip install git+https://github.com/google-deepmind/tapnet.git
pip install git+https://github.com/google-deepmind/recurrentgemma.git@main
pip install pytorch_lightning==2.4.0 einops moviepy prettytable           # AllTracker

# 3. Model weights
echo "⬇️  [3/5] Model weights"
mkdir -p third_party/s2m2/weights/pretrain_weights third_party/co-tracker/weights \
         third_party/tapnext_weights third_party/sam_weights \
         third_party/alltracker/weights

wget -nc -O third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth    "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/co-tracker/weights/cotracker3_offline.pth       "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
wget -nc -O third_party/tapnext_weights/tapnextpp_512.ckpt             "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth               "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
wget -nc -O third_party/alltracker/weights/alltracker.pth              "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

echo "  ℹ️  VGGT: downloaded on first use"

# 4. Verify
echo "🔍 [4/5] Verify checkpoints"
for f in third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth \
         third_party/co-tracker/weights/cotracker3_offline.pth \
         third_party/tapnext_weights/tapnextpp_512.ckpt \
         third_party/sam_weights/sam_vit_h_4b8939.pth \
         third_party/alltracker/weights/alltracker.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

# 5. Done
echo "🎉 [5/5] Setup complete"
