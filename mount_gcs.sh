#!/bin/bash
# GCS 容器网络盘/数据集一键挂载脚本

echo "========================================="
echo "1. 挂载 DROID 官方源视频与轨迹数据集"
echo "========================================="
mkdir -p ~/droid_data/input/robotics/droid_raw
gcsfuse --implicit-dirs --only-dir robotics/droid_raw gresearch ~/droid_data/input/robotics/droid_raw
echo "✅ 输入数据集挂载成功."

echo "========================================="
echo "2. 挂载推断产出/多视角深度专用存储桶"
echo "========================================="
mkdir -p ~/droid_data/output/mv-tap
gcsfuse --implicit-dirs --only-dir mv-tap dm-tapnet ~/droid_data/output/mv-tap
echo "✅ 输出存储桶挂载成功."

echo "🚀 双端数据流打通，随时可运行流水线！"
