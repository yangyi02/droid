#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "📦 [1/5] Submodules"
git submodule update --init --recursive

echo "🐍 [2/5] Python packages"
pip install -r requirements.txt

echo "⚡ [3/5] PyBullet NumPy support"
if python -c "import pybullet, sys; sys.exit(0 if pybullet.isNumpyEnabled() else 1)" 2>/dev/null; then
  echo "  ✅ pybullet already built with NumPy support"
else
  echo "  ⏳ Rebuilding pybullet from source, takes several minutes..."
  pip install --force-reinstall --no-deps --no-binary pybullet \
      --no-build-isolation --no-cache-dir pybullet
  python -c "import pybullet; print('  isNumpyEnabled =', pybullet.isNumpyEnabled())"
fi

echo "⬇️  [4/5] Model weights"
mkdir -p third_party/s2m2/weights third_party/sam_weights

wget -nc -O third_party/s2m2/weights/CH384NTR3.pth        "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo "🔍 [5/5] Verify weights"
for f in third_party/s2m2/weights/CH384NTR3.pth \
         third_party/sam_weights/sam_vit_h_4b8939.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

echo "🎉 Setup complete"
