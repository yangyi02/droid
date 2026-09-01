#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules
echo "📦 [1/5] Submodules"
git submodule update --init --recursive

# 2. Python packages
echo "🐍 [2/5] Python packages"
pip install -r requirements.txt


# 3. PyBullet, rebuilt with NumPy support
echo "⚡ [3/5] PyBullet NumPy support"
# PyPI ships no Linux wheel for pybullet, so pip always builds it from the
# sdist -- and by default it builds inside an isolated environment that has no
# numpy. setup.py probes for numpy with a bare `try: import numpy`, so in that
# environment it silently drops NumPy support (it prints "numpy is disabled",
# which pip hides). Without it getCameraImage marshals every pixel into a
# Python tuple before handing it back: a 1280x720 render costs ~380 ms instead
# of ~23 ms, and the 3.7M-element RGB tuple we never use is over half of that.
if python -c "import pybullet, sys; sys.exit(0 if pybullet.isNumpyEnabled() else 1)" 2>/dev/null; then
  echo "  ✅ pybullet already built with NumPy support"
else
  echo "  ⏳ Rebuilding pybullet from source, takes several minutes..."
  #   --no-build-isolation : let the build import the numpy installed above
  #   --no-cache-dir       : never reuse a previously built non-NumPy wheel
  pip install --force-reinstall --no-deps --no-binary pybullet \
      --no-build-isolation --no-cache-dir pybullet
  python -c "import pybullet; print('  isNumpyEnabled =', pybullet.isNumpyEnabled())"
fi

# 4. Model weights
echo "⬇️  [4/5] Model weights"
mkdir -p third_party/s2m2/weights third_party/sam_weights

wget -nc -O third_party/s2m2/weights/CH384NTR3.pth        "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# 5. Verify
echo "🔍 [5/5] Verify weights"
for f in third_party/s2m2/weights/CH384NTR3.pth \
         third_party/sam_weights/sam_vit_h_4b8939.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

# 6. Done
echo "🎉 Setup complete"
