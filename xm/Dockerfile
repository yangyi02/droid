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
 && pip install -r /tmp/requirements-depth.txt

RUN cd /tmp && python /usr/local/zed/get_python_api.py && rm -f /tmp/*.whl \
 && pip install --force-reinstall --no-deps requests urllib3 charset-normalizer idna certifi

RUN pip install --force-reinstall --no-deps --no-binary pybullet --no-build-isolation pybullet

ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

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

RUN mkdir -p /usr/local/zed/settings \
 && grep -oE '"[0-9]+"' data/meta/1.0.1/camera_serials.json data/meta/1.0.1/cam2base_extrinsic_superset.json \
  | cut -d'"' -f2 | sort -u \
  | while read -r sn; do \
      wget -q -O "/usr/local/zed/settings/SN${sn}.conf" "https://calib.stereolabs.com/?SN=${sn}" || true; \
      [ "$(stat -c %s "/usr/local/zed/settings/SN${sn}.conf" 2>/dev/null || echo 0)" -ge 500 ] || rm -f "/usr/local/zed/settings/SN${sn}.conf"; \
    done \
 && chmod -R a+rwX /usr/local/zed

COPY . .

RUN setpriv --reuid=65534 --regid=65534 --clear-groups bash -c "find /app /usr/local /usr/lib/python3.10 \
      /usr/lib/x86_64-linux-gnu /usr/bin -type f -readable -print0 2>/dev/null | xargs -0 md5sum" > /checksums.md5

ENTRYPOINT ["bash", "-c", \
  "md5sum --quiet --strict -c /checksums.md5 && exec \"$@\"; echo 'Corrupted image, holding GPU'; sleep 6h; exit 1", \
  "--", "python", "-u"]
