#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules
echo "📦 [1/4] Submodules"
git submodule update --init --recursive

# 2. Python packages
echo "🐍 [2/4] Python packages"
pip install -r requirements.txt


# 3. Model weights
echo "⬇️  [3/4] Model weights"
mkdir -p third_party/s2m2/weights/pretrain_weights third_party/sam_weights

wget -nc -O third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth    "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth               "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# 4. Verify
echo "🔍 [4/4] Verify checkpoints"
for f in third_party/s2m2/weights/pretrain_weights/CH384NTR3.pth \
         third_party/sam_weights/sam_vit_h_4b8939.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

# 5. Done
echo "🎉 Setup complete"
