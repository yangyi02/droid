#!/bin/bash
# DROID Pipeline — one-time setup. Usage: bash setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Submodules
echo "📦 [1/4] Submodules"
git submodule update --init --recursive

# s2m2 is imported in place and given its weights in place, so both the
# bytecode caches and the 1.6 GB checkpoint land inside the submodule. Upstream
# ships no .gitignore, so they show up as untracked in the submodule's OWN
# status -- which is what editors colour, and what the parent's .gitignore
# cannot reach, since ignore rules do not cross into a submodule. Exclude them
# where the submodule itself reads them, touching nothing upstream tracks.
# --git-path resolves whether .git there is a directory or a gitdir pointer.
exclude="third_party/s2m2/$(git -C third_party/s2m2 rev-parse --git-path info/exclude)"
for rule in '__pycache__/' '/weights/'; do
    grep -qxF "$rule" "$exclude" 2>/dev/null || echo "$rule" >> "$exclude"
done

# 2. Python packages
echo "🐍 [2/4] Python packages"
pip install -r requirements.txt


# 3. Model weights
echo "⬇️  [3/4] Model weights"
mkdir -p third_party/s2m2/weights third_party/sam_weights

wget -nc -O third_party/s2m2/weights/CH384NTR3.pth        "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth"
wget -nc -O third_party/sam_weights/sam_vit_h_4b8939.pth  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# 4. Verify
echo "🔍 [4/4] Verify weights"
for f in third_party/s2m2/weights/CH384NTR3.pth \
         third_party/sam_weights/sam_vit_h_4b8939.pth; do
  [ -f "$f" ] && echo "  ✅ $f ($(du -h "$f" | cut -f1))" || echo "  ❌ MISSING: $f"
done

# 5. Done
echo "🎉 Setup complete"
