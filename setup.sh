#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v ffmpeg >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y ffmpeg
fi

pip install -r requirements.txt

if ! python -c "import pybullet, sys; sys.exit(0 if pybullet.isNumpyEnabled() else 1)" 2>/dev/null; then
  pip install --force-reinstall --no-deps --no-binary pybullet \
      --no-build-isolation --no-cache-dir pybullet
fi

if [ "$1" = "--no-depth" ]; then
  exit 0
fi

git submodule update --init --recursive
pip install -r requirements-depth.txt

if ! command -v ZED_Explorer >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y zstd
  wget -nc -O ZED_SDK_Linux_Ubuntu22.run "https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22"
  chmod +x ZED_SDK_Linux_Ubuntu22.run
  ./ZED_SDK_Linux_Ubuntu22.run silent runtime_only skip_tools
  find /usr/local/zed -name "pyzed*.whl" -exec pip install {} +
  pip install numpy==2.0.2
fi

mkdir -p third_party/s2m2/weights third_party/segment_anything/weights
wget -nc -O third_party/s2m2/weights/CH384NTR3.pth "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/segment_anything/weights/sam_vit_h_4b8939.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
