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

CUDNN_LIB="$(python -c 'import os, site; print(os.path.join(site.getsitepackages()[0], "nvidia", "cudnn", "lib"))')"
if [ -n "${VIRTUAL_ENV:-}" ] && ! grep -q "nvidia/cudnn/lib" "$VIRTUAL_ENV/bin/activate"; then
  echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB:\$LD_LIBRARY_PATH\"" >> "$VIRTUAL_ENV/bin/activate"
fi

if [ "$1" = "--no-depth" ]; then
  exit 0
fi

git submodule update --init --recursive
pip install -r requirements-depth.txt

if ! python -c "import pyzed.sl" 2>/dev/null; then
  (
    cd "$(mktemp -d)"
    if [ ! -f /usr/local/zed/get_python_api.py ]; then
      sudo apt-get update -qq && sudo apt-get install -y zstd
      wget -O zed_sdk.run "https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22"
      chmod +x zed_sdk.run
      ./zed_sdk.run silent runtime_only skip_tools
    fi
    python /usr/local/zed/get_python_api.py
  )
fi

mkdir -p third_party/s2m2/weights third_party/segment_anything/weights
wget -nc -P third_party/s2m2/weights "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -P third_party/segment_anything/weights "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
