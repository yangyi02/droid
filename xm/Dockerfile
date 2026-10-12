FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git wget ca-certificates \
      ffmpeg zstd lsb-release sudo udev libusb-1.0-0 libgomp1 libpng16-16 \
      libegl1 libgles2 libglvnd0 libopengl0 libgl1 \
      libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN wget -q -O /tmp/zed_sdk.run "https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22" \
 && chmod +x /tmp/zed_sdk.run \
 && /tmp/zed_sdk.run -- silent runtime_only skip_tools skip_cuda \
 && ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so \
 && rm -f /tmp/zed_sdk.run

COPY requirements.txt requirements-depth.txt /tmp/
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt \
 && pip install -r /tmp/requirements-depth.txt \
 && pip install requests google-cloud-storage

RUN cd /tmp && python /usr/local/zed/get_python_api.py && rm -f /tmp/*.whl

RUN pip install --force-reinstall --no-deps --no-binary pybullet --no-build-isolation pybullet \
 && python -c "import pybullet, importlib.util as u; assert pybullet.isNumpyEnabled(); assert u.find_spec('eglRenderer')"

RUN pip install --force-reinstall --no-deps requests urllib3 charset-normalizer idna certifi \
 && python -c "from google.cloud import storage; from google.cloud.storage import transfer_manager; import requests.adapters"

ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton \
    XDG_CACHE_HOME=/tmp/.cache

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

WORKDIR /app

ARG S2M2_COMMIT=8f3bd785df593619d955203c6f2c741bef6e31df
RUN git clone https://github.com/junhong-3dv/s2m2.git third_party/s2m2 \
 && git -C third_party/s2m2 checkout "${S2M2_COMMIT}" \
 && mkdir -p third_party/s2m2/weights third_party/segment_anything/weights \
 && wget -q -P third_party/s2m2/weights "https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth" \
 && wget -q -P third_party/segment_anything/weights "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

RUN mkdir -p data/meta/1.0.1 \
 && for f in camera_serials.json episode_id_to_path.json cam2base_extrinsic_superset.json; do \
      wget -q -O "data/meta/1.0.1/$f" "https://huggingface.co/KarlP/droid/resolve/main/$f"; \
    done

RUN python -c "import json; m='data/meta/1.0.1/'; d=json.load(open(m+'camera_serials.json')); x=json.load(open(m+'cam2base_extrinsic_superset.json')); s={str(v) for e in d.values() for v in e.values()} | {str(k) for e in x.values() for k in e}; print(chr(10).join(sorted(v for v in s if v.isdigit())))" > /tmp/serials.txt \
 && mkdir -p /usr/local/zed/settings \
 && while read -r sn; do \
      f="/usr/local/zed/settings/SN${sn}.conf"; \
      wget -q -O "$f" "https://calib.stereolabs.com/?SN=${sn}" || true; \
      if [ ! -s "$f" ] || [ "$(stat -c %s "$f")" -lt 500 ]; then echo "no calibration for ${sn}"; rm -f "$f"; fi; \
    done < /tmp/serials.txt \
 && echo "ZED calibrations: $(ls /usr/local/zed/settings | wc -l) / $(wc -l < /tmp/serials.txt) serials" \
 && chmod -R a+rwX /usr/local/zed
ENV ZED_REQUIRE_LOCAL_CALIB=1

COPY assets ./assets
COPY core ./core
COPY tapvidmv ./tapvidmv
COPY config.py compute_depth.py compute_extrinsics.py compute_scene.py compute_tracks.py \
     compute_metrics.py compute_review.py episodes_success.txt ./
COPY xm/run_stage.py ./xm/run_stage.py
COPY xm/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
