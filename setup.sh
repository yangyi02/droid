#!/bin/bash
# DROID Pipeline — One-time setup
# Usage: bash setup.sh
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$REPO/third_party"

# ── Helpers ──────────────────────────────────────────────────

log()  { echo "  $1"; }
step() { echo ""; echo "[$1] $2"; }

need_download() {
  local f="$1" min="${2:-1000}"
  [[ ! -f "$f" ]] || [[ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -lt "$min" ]]
}

hf_get() {
  local repo="$1" file="$2" dir="$3"
  local dest="$dir/$file"
  if need_download "$dest"; then
    log "↓ $file"
    rm -f "$dest"
    hf download "$repo" "$file" --local-dir "$dir" --quiet
    log "✓ $file ($(du -h "$dest" | cut -f1))"
  else
    log "· $file (cached)"
  fi
}

wget_get() {
  local url="$1" dest="$2"
  if need_download "$dest"; then
    log "↓ $(basename "$dest")"
    wget --progress=bar:force -O "$dest" "$url"
    log "✓ $(basename "$dest") ($(du -h "$dest" | cut -f1))"
  else
    log "· $(basename "$dest") (cached)"
  fi
}

# ── 1. Submodules ────────────────────────────────────────────

step "1/4" "Git submodules"
cd "$REPO" && git submodule update --init --recursive
log "✓ ready"

# ── 2. Python packages ──────────────────────────────────────

step "2/4" "Python dependencies"
pip install -q \
    pybullet opencv-python scipy tqdm h5py \
    mediapy yourdfpy plotly pyrender \
    huggingface_hub hf_transfer
pip install -q git+https://github.com/facebookresearch/segment-anything.git
pip install -q "$TP/co-tracker"
pip install -q git+https://github.com/google-deepmind/tapnet.git 2>/dev/null || true
pip install -q git+https://github.com/google-deepmind/recurrentgemma.git@main 2>/dev/null || true
log "✓ installed"

# ── 3. Model weights ────────────────────────────────────────

step "3/4" "Model weights"
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$TP/s2m2/weights/pretrain_weights" \
         "$TP/co-tracker/weights" \
         "$TP/tapnext_weights" \
         "$TP/sam_weights"

# HuggingFace models (via huggingface-cli, the only method that handles xet CDN)
hf_get  minimok/s2m2         CH384NTR3.pth       "$TP/s2m2/weights/pretrain_weights"
hf_get  facebook/cotracker3  scaled_offline.pth   "$TP/co-tracker/weights"

# CoTracker expects a specific filename
CT_SRC="$TP/co-tracker/weights/scaled_offline.pth"
CT_DST="$TP/co-tracker/weights/cotracker3_offline.pth"
[[ -f "$CT_SRC" && ! -f "$CT_DST" ]] && cp "$CT_SRC" "$CT_DST"

# Direct downloads (wget works fine for non-HF hosts)
wget_get "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt" \
         "$TP/tapnext_weights/tapnextpp_512.ckpt"
wget_get "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
         "$TP/sam_weights/sam_vit_h_4b8939.pth"

log "· VGGT: downloaded on first use"
log "✓ all weights ready"

# ── 4. Done ──────────────────────────────────────────────────

step "4/4" "Ready"
cat <<'EOF'
  droid/
  ├── pipeline.ipynb          ← start here
  ├── compute_depth.py        Stage 1: stereo depth
  ├── compute_extrinsics.py   Stage 2: camera calibration
  ├── compute_tracks.py       Stage 3: 3D tracking
  ├── run_parallel.sh         multi-GPU runner
  └── run_extrinsics_ablation.py  ablation experiments
EOF
