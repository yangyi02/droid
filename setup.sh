#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

WITH_DEPTH=1
if [ "$1" = "--no-depth" ]; then
  WITH_DEPTH=0
fi

echo "🎬 System packages"
if command -v ffmpeg >/dev/null; then
  echo "  ✅ ffmpeg already installed"
else
  echo "  ⏳ Installing ffmpeg (mediapy decodes the episode videos through it)..."
  sudo apt-get update -qq && sudo apt-get install -y ffmpeg
fi

echo "🐍 Python packages"
pip install -r requirements.txt

echo "⚡ PyBullet NumPy support"
if python -c "import pybullet, sys; sys.exit(0 if pybullet.isNumpyEnabled() else 1)" 2>/dev/null; then
  echo "  ✅ pybullet already built with NumPy support"
else
  echo "  ⏳ Rebuilding pybullet from source, takes several minutes..."
  pip install --force-reinstall --no-deps --no-binary pybullet \
      --no-build-isolation --no-cache-dir pybullet
  python -c "import pybullet; print('  isNumpyEnabled =', pybullet.isNumpyEnabled())"
fi

if [ "$WITH_DEPTH" = 0 ]; then
  echo "⏭️  Skipping Stage 1 (--no-depth): s2m2, Segment Anything, ZED SDK, model weights"
  echo "🎉 Setup complete"
  exit 0
fi

echo "📦 Stage 1 submodule"
git submodule update --init --recursive

echo "🐍 Stage 1 Python packages"
pip install -r requirements-depth.txt

echo "📷 Stage 1 ZED SDK"
if command -v ZED_Explorer >/dev/null; then
  echo "  ✅ ZED SDK already installed"
else
  echo "  ⏳ Installing ZED SDK (runtime only), takes a few minutes..."
  sudo apt-get update -qq && sudo apt-get install -y zstd
  wget -nc -O ZED_SDK_Linux_Ubuntu22.run "https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22"
  chmod +x ZED_SDK_Linux_Ubuntu22.run
  ./ZED_SDK_Linux_Ubuntu22.run silent runtime_only skip_tools
  find /usr/local/zed -name "pyzed*.whl" -exec pip install {} +
  pip install numpy==2.0.2
fi

echo "⬇️  Stage 1 model weights"
mkdir -p third_party/s2m2/weights third_party/segment_anything/weights

wget -nc -O third_party/s2m2/weights/CH384NTR3.pth        "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/segment_anything/weights/sam_vit_h_4b8939.pth  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo "🔍 Verify weights"
for f in third_party/s2m2/weights/CH384NTR3.pth \
         third_party/segment_anything/weights/sam_vit_h_4b8939.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

echo "🎉 Setup complete"
