#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules (CoTracker, S2M2, etc.)
echo "📦 [1/6] Submodules"
git submodule update --init --recursive

# 2. Clone additional tracker repos
echo "📦 [2/6] Clone AllTracker & Track-On-R"
if [ ! -d "third_party/alltracker" ]; then
  git clone https://github.com/aharley/alltracker.git third_party/alltracker
else
  echo "  ✓ third_party/alltracker already exists"
fi
if [ ! -d "third_party/track_on" ]; then
  git clone https://github.com/gorkaydemir/track_on.git third_party/track_on
else
  echo "  ✓ third_party/track_on already exists"
fi

# 3. Python packages
echo "🐍 [3/6] Python packages"
pip install pybullet opencv-python scipy tqdm h5py mediapy yourdfpy plotly pyrender
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install third_party/co-tracker
pip install git+https://github.com/google-deepmind/tapnet.git
pip install git+https://github.com/google-deepmind/recurrentgemma.git@main

# AllTracker deps (skip torch/torchvision/torchaudio — already installed)
echo "  🔧 AllTracker dependencies"
pip install pytorch_lightning==2.4.0 einops moviepy prettytable

# Track-On-R deps
echo "  🔧 Track-On-R dependencies"
pip install timm transformers accelerate av einshape dm-tree
# mmcv requires CUDA — install prebuilt wheel for CUDA 12.1 + PyTorch 2.4
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html 2>/dev/null \
  || echo "  ⚠️  mmcv prebuilt wheel failed — you may need to build from source (see Track-On-R README)"

# 4. Model weights
echo "⬇️  [4/6] Model weights"
mkdir -p third_party/s2m2/weights/pretrain_weights \
         third_party/co-tracker/weights \
         third_party/tapnext_weights \
         third_party/sam_weights \
         third_party/alltracker/weights \
         third_party/track_on/weights

wget -nc -O third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth    "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/co-tracker/weights/cotracker3_offline.pth       "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
wget -nc -O third_party/tapnext_weights/tapnextpp_512.ckpt             "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth               "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
wget -nc -O third_party/alltracker/weights/alltracker.pth              "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
wget -nc -O third_party/track_on/weights/track_on_r.pt                 "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt"

echo "  ℹ️  VGGT: downloaded on first use"
echo "  ℹ️  Track-On-R DINOv3 backbone: auto-downloaded on first use (requires HF login)"

# 5. Verify
echo "🔍 [5/6] Verify checkpoints"
for f in \
  third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth \
  third_party/co-tracker/weights/cotracker3_offline.pth \
  third_party/tapnext_weights/tapnextpp_512.ckpt \
  third_party/sam_weights/sam_vit_h_4b8939.pth \
  third_party/alltracker/weights/alltracker.pth \
  third_party/track_on/weights/track_on_r.pt; do
  if [ -f "$f" ]; then
    sz=$(du -h "$f" | cut -f1)
    echo "  ✅ $f ($sz)"
  else
    echo "  ❌ MISSING: $f"
  fi
done

# 6. Done
echo "🎉 [6/6] Setup complete"
