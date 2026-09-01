#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
# Find the repo root by walking up from this script to the directory holding
# core/io.py — the very file core.io.DATA_ROOT anchors on, so the shell and
# Python can never disagree about where droid_data/ is. Anchoring on a marker
# instead of a fixed "../" also means this keeps working if the script is moved
# into a subdirectory. (Inlined rather than shared: bootstrap code that locates
# the repo cannot itself live inside the repo it has yet to locate.)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -f "$REPO_ROOT/core/io.py" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -f "$REPO_ROOT/core/io.py" ]; then
    echo "❌ Not inside the droid repo: no core/io.py found above ${BASH_SOURCE[0]}" >&2
    exit 1
fi
cd "$REPO_ROOT"

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
