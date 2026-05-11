# droid pipeline vs PointWorld pipeline 详尽对比

> **两套代码的关系**
> - **你自己的 droid pipeline**：`droid/process_droid_stage1.py` + `process_droid_stage2.py` + `run_parallel.sh`
> - **NVIDIA PointWorld pipeline**：`droid/PointWorld/real/` 下的全部脚本（arXiv 2601.03782）

---

## 一、定位与目标

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **核心目标** | 为 DROID 数据集做相机外参标定 | 生成完整 3D 点流训练数据（外参 + 深度 + 2D 追踪 → 3D flows） |
| **输出用途** | 得到高质量 extrinsics，供下游使用 | 直接生成 PointWorld 模型的训练/测试 WDS shards |
| **数据域** | DROID（ZED SVO raw） | DROID（real） + BEHAVIOR-1K（sim，OmniGibson） |
| **代码规模** | ~1200 行（2 文件） | ~15000+ 行（20+ 模块） |
| **代码风格** | 单文件 monolithic，流水线内联 | 模块化，类封装，接口清晰 |
| **作者** | 个人研究代码 | NVIDIA 生产级开源（Apache-2.0） |

---

## 二、整体架构

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **Stage 划分** | Stage 1（深度提取）→ Stage 2（外参优化） | compute_depth → compute_extrinsics → filter → compute_2d_flows → convert_2d_flows_to_3d → integrity_check → make_manifest → convert_wds |
| **阶段耦合** | Stage 2 直接读取 Stage 1 的 `.npz` | 每个脚本独立，通过文件路径约定解耦 |
| **状态传递** | 内存中 `scene_constants` 字典贯穿全程 | 每阶段读写 HDF5 文件，无跨阶段内存共享 |
| **错误处理** | `try/except` 跳过失败 episode | 每脚本有 fast-fail 检查 + `debug` 模式立即抛异常 |
| **进度追踪** | 文件锁追加 `episodes_stage{1,2}.txt` | 跳过已存在输出文件（幂等重跑） |
| **并发模型** | 单机多卡（`gnu parallel`） | 无状态分片（`--rank / --world_size`），任意多节点 |

---

## 三、数据输入

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **原始数据格式** | ZED `.svo` 文件 + `trajectory.h5` + `metadata_*.json` | DROID raw GCS 路径（`gs://xembodiment_data/r2d2/...`） |
| **数据加载方式** | 本地路径，需预先下载到 `~/droid_data/input/` | 流式读取 GCS 或强制本地缓存（`POINTWORLD_CACHE_DIR`） |
| **Episode 发现** | 从 HuggingFace `KarlP/droid` 下载 `episode_id_to_path.json` | 从 `real/droid_paths.txt` 读取 GCS 路径列表 |
| **有效帧过滤** | 从 `keep_ranges_1_0_1.json` 读取有效时间段 | 用 `ee_pos_threshold` / `ee_rot_threshold` 过滤静止帧 |
| **时间对齐** | 按最短流长度截断（robot/camera 流对齐） | 按 canonical timestamps 做 nearest-neighbor 匹配 |
| **Wrist 相机处理** | 和外部相机同样解码，单独做 gripper 修复 | 读取前先做 `cv2.ROTATE_180`（补偿相机装反） |
| **元数据来源** | HuggingFace `camera_serials.json`, `cam2base_extrinsic_superset.json` | DROID scene 的 `metadata.json` + `gripper2wrist_transforms.json` |

---

## 四、深度估计

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **使用模型** | **S2M2 XL**（junhong-3dv/s2m2，立体匹配） | **FoundationStereo**（NVlabs，生产级） |
| **深度类型** | Stereo disparity → metric depth | Stereo disparity → metric depth（更鲁棒） |
| **辅助模块** | 无（S2M2 独立） | 内嵌 **DINOv2**（特征）+ **Depth-Anything**（单目辅助） |
| **模型加载** | `load_model(weights_path, "XL")` + `torch.compile` | `FoundationStereo_inference(cfg, ckpt)` |
| **模型加速** | `torch.compile`（编译加速） | 无编译（依赖模型本身优化） |
| **置信度过滤** | ✅ `conf < 0.95` 的视差置零 | 无显式置信度过滤 |
| **Gripper 区域修复** | ✅ SAM mask + temporal median → 深度注入 | ❌ 不处理 |
| **输出格式** | `raw_depth.npz`（uint16 mm）+ `original_raw_depth.npz`（wrist 备份） | `{uuid}_depth.h5`（包含 timestamps + uint16 mm） |
| **运行方式** | Stage 1 中串行处理（每 episode 按帧循环） | `compute_depth.py` 独立脚本，`--rank/--world_size` 分布式 |

### Gripper 深度修复（你独有）

```
阶段 1：SAM ViT-H 提取 gripper mask
  → 仅处理 gripper_positions < 0.05（闭合帧）
  → 每帧调用 SAM（点 prompt + box prompt）→ 选面积合理的最优 mask

阶段 2：Temporal Median 蒸馏
  → 收集所有闭合帧在 mask 区域内的有效深度值（depth_bank: N×H×W）
  → np.nanmedian(depth_bank, axis=0) → 单帧净化深度图

阶段 3：深度注入
  → 仅在 valid_mask > 0 的像素上替换原始立体深度
  → 结果写入 raw_depth.npz，原始备份写入 original_raw_depth.npz
```

---

## 五、相机外参初始化（VGGT 使用方式）

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **VGGT 加载** | `VGGT.from_pretrained("facebook/VGGT-1B")`（HuggingFace 自动下载） | `torch.load(vggt_model_path)`（本地 `model.pt`） |
| **送入帧数** | 仅第一帧（每相机 1 张图） | 多帧全部送入（N 帧 × M 相机） |
| **帧序顺序** | `[ext1_frame0, ext2_frame0, ..., wrist_frame0]` | `[ext1所有帧, wrist所有帧, ext2所有帧, ...]` |
| **Pose averaging** | ❌ 无，只用第一帧结果 | ✅ 对每个相机的多帧估计做 SO(3) 平均（`average_poses`） |
| **VGGT world frame 锚定** | 链式：`T_ref_to_base = T_wrist_to_base_first @ inv(T_ref_to_wrist)` | 对每个 wrist 帧分别算 `T_base_ext0 = inv(T_ext0_wrist) @ T_base_wrist(t)`，再平均 |
| **内参处理** | 使用 ZED 物理内参，VGGT 结果仅用于外参 | VGGT 同时输出 `vggt_intrinsics`，作为参考（保留 measured 和 vggt 两份） |
| **输出格式** | 内存中的 `scene_state` 字典 | `cameras.json`（包含 `vggt_extrinsics` 和 `optimized_extrinsics` 两份归档） |
| **精度** | 依赖单帧，对遮挡/模糊敏感 | 多帧平均，更鲁棒 |

---

## 六、机械臂渲染

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **渲染引擎** | **PyBullet**（OpenGL EGL headless） | **urdfpy + trimesh**（纯 CPU，无 OpenGL） |
| **URDF 数量** | 两个 URDF body（dual-body trick） | 单个 URDF |
| **URDF 来源** | body1: `pybullet_data` 自带 `franka_panda/panda.urdf`；body2: PointWorld `franka_panda_robotiq_2f85_og.urdf` | PointWorld `franka_panda_robotiq_2f85_og.urdf` |
| **双 URDF 原因** | PyBullet 自带 Panda 无 Robotiq，PointWorld URDF 有 Robotiq 但 Panda 几何不精确，两者互补 | 单 URDF 已包含完整 Panda + Robotiq |
| **Gripper 驱动** | `finger_joint` angle = `clip(g,0,1)*0.8028 - offset`，带 sign 翻转 | `finger_joint` 直接传入 urdfpy FK config |
| **FK 实现** | PyBullet 内置 `resetJointState` + `performCollisionDetection` | `urdfpy.URDF.visual_trimesh_fk(cfg)` |
| **FK 缓存** | ❌ 无缓存 | ✅ `fk_cache[round_key]` 避免重复计算相同关节角 |
| **渲染输出** | 完整深度图（H×W，每像素精确深度）+ 分割图（object+link id） | 表面采样点云（总 25000 点，按面积比例分配） |
| **采样策略** | N/A（光栅化覆盖所有像素） | 面积比例分配；gripper 部件（finger/knuckle/robotiq）额外 ×2 权重；hand_camera_part 面积 ×1e-6 抑制 |
| **点云缓存** | ❌ 无缓存 | ✅ `world_points_cache[fk_pose_key]` |
| **投影矩阵** | K → OpenGL proj matrix（手工公式推导） | K → 直接矩阵乘法投影到 image coords |
| **分割/前景提取** | `seg_buffer & 0xFFFFFF == ghost_id` 提取 gripper 像素 | 无 seg，通过 gripper 关键字过滤采样点 |
| **用途** | 生成渲染深度图 + 提取前景点云（用于 loss 计算） | 生成世界坐标点云（直接参与 loss 计算） |
| **速度** | 需要 EGL GPU 环境，但渲染完整帧速度快 | CPU 友好，无 EGL 依赖，但点密度受采样数限制 |

---

## 七、相机外参优化

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **优化框架** | PyTorch autograd + `optim.Adam` | PyTorch autograd + `optim.Adam` 或 `optim.LBFGS` |
| **优化变量** | 6D 轴角+平移 `d_ext∈ℝ⁶`，直接优化 | `translation_params_norm`、`rotation_params_norm` 各 ∈ℝ³，归一化后映射到物理量 |
| **参数归一化** | ❌ 无 scale 归一化 | ✅ `t = t_norm * translation_scale`（默认 0.01m）；`r = r_norm * rotation_scale`（默认 1°） |
| **施加方式** | Left-multiply：`T_final = T_init @ make_delta_T(d_ext)` | Left-multiply：`T_opt = pose_6dof_to_matrix(Δ) @ T_init` |
| **旋转参数化** | Rodrigues 轴角（带 Taylor fallback 保证 θ→0 数值稳定） | ZYX Euler 角（roll/pitch/yaw，直接三角函数） |
| **损失函数** | Depth re-projection L1（`|Z_obs - Z_pred|`） | Depth re-projection L1（同）|
| **深度采样方式** | `F.grid_sample`（bilinear，batch） | `sample_depth_with_grid_sample`（封装的 `F.grid_sample`，batch） |
| **有效深度范围** | `0 < Z < 1.5 m` | `MIN_DEPTH_M=0.3 m < Z < MAX_DEPTH_M=2.0 m` |
| **去重/稀疏化** | ❌ 无 | ✅ `deduplicate_coordinates`（坐标量化到 grid cell，保留每 cell 首次击中） |
| **机器人可见性预检** | ❌ 无，直接优化 | ✅ 优化前过滤：`valid_count < min_robot_points`（默认 1000）的帧剔除 |
| **相机联合优化** | ✅ Stage 3/4 所有相机同时优化 | ❌ 各外部相机独立优化 |
| **全局一致性** | ✅ Chamfer loss（3 视角背景点云互相约束，5cm 截断） | ❌ 无跨相机一致性约束 |
| **wrist 手眼矩阵** | ✅ Stage 2b 独立优化 `T_cam_ee`（相机相对 EE 的固定变换） | ❌ wrist 相机不参与优化，只由运动学 GT 决定 |
| **Dual-base 竞争** | ✅ VGGT 初始化 vs 数据集初始化各跑一次，选 loss 低的 | ❌ 仅用 VGGT 初始化 |
| **优化阶段数** | 5阶段（Stage 0 → 1 → 2a → 2b → 3 → 4） | 1阶段（VGGT init → joint optimize） |
| **学习率策略** | Stage 3 lr=0.001 (500步) → Stage 4 lr=0.0001 (500步) | 单阶段，默认 lr=0.05，max_iter=2000 |
| **最终输出** | `extrinsics.npz`（所有相机的 base_extrinsic + per-frame trajectory） | `cameras.json`（vggt + optimized extrinsics，所有相机，含 intrinsics） |

### 你的外参优化 5 阶段流详解

```
Stage 0  从数据集 cam2base_extrinsic_superset.json 读取预标定外参（如存在）
           → init_scene_state

Stage 1  VGGT 视觉锚定：
           img_list = [ext1_frame0, ext2_frame0, ..., wrist_frame0]
           T_rel = VGGT(img_list)
           T_ref_to_base = T_wrist_to_base_first @ inv(T_ref_to_wrist)  # 链式转换
           → vggt_scene_state

Stage 2a 外部相机对齐（dual-base 竞争）：
           for each ext_cam:
             candidates = [("VGGT", T_vggt), ("Dataset", T_dataset)]
             for each candidate:
               5 outer loops × 100 inner steps（每outer随机采样100帧）
               loss = depth_reprojection_L1(render_pts, T_opt, K, depth_obs)
             选 loss 最低的结果
           → pybullet_scene_state（外部相机）

Stage 2b 腕部相机手眼标定：
           优化 T_cam_ee（相机相对末端执行器的固定变换）
           5 outer × 100 inner，用 gripper 分割前景点
           loss = depth_reprojection_L1(P_ee_anchored, T_cam_ee_opt, K, wrist_depth)
           → pybullet_scene_state（wrist 相机）

Stage 3  全局联合优化（粗）：
           同时优化 d1,d2,dhe（三个相机的6D delta pose）
           loss = loss_chamfer + 1.0*(loss_rob1 + loss_rob2 + loss_wrist)
           lr=0.001, 500 步，预缓存所有帧点云

Stage 4  精调：
           相同结构，lr=0.0001, 500 步，robot_weight=0.1
```

---

## 八、2D 点追踪

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **是否使用** | 加载了 CoTracker3 模型但 Stage 1/2 主流程未调用 | ✅ 核心步骤（`compute_2d_flows.py`） |
| **模型** | CoTracker3 (`cotracker3_offline.pth`) | CoTracker3 (`scaled_online.pth`) |
| **追踪模式** | offline（加载全序列） | online（逐帧） |
| **追踪粒度** | N/A | 密集轨迹 (T, N, 2) + visibility (T, N) |
| **种子点策略** | N/A | workspace mask & ~robot mask 区域内的规则网格点 |
| **机械臂遮罩** | SAM gripper mask（仅闭合帧，静态） | robot FK mask（每帧动态，urdfpy + FK） |
| **workspace 约束** | N/A | ConvexHull 投影 workspace bounds，只追踪 workspace 内的点 |
| **输出格式** | N/A | `{uuid}_2d_flows.h5`（含 flows_2d_xy, visibility, flow_colors, proprio） |
| **追踪质量过滤** | N/A | outlier 去除（`remove_outlier_flows`，eps=0.02~0.05m，DBSCAN 类方法） |

---

## 九、3D 点流生成

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **是否实现** | ❌ 无，pipeline 到外参标定为止 | ✅ `convert_2d_flows_to_3d.py`，`Flow2DTo3DConverter` |
| **输入** | N/A | 2D 轨迹 .h5 + depth .h5 + cameras.json |
| **核心方法** | N/A | `z = depth[t, y_int, x_int]`，反投影：`P_cam = [(x-cx)/fx*z, (y-cy)/fy*z, z]`，`P_world = cam2world @ P_cam` |
| **深度采样优化** | N/A | 逐帧流式采样（只读 track 坐标处的 depth，不分配完整 T×H×W） |
| **法向量估计** | N/A | Open3D `estimate_normals`（KDTree radius=0.1m，max_nn=30）+ 朝向相机翻转 |
| **法向量存储** | N/A | int8 量化（`np.rint(n*127).astype(int8)`，4× 压缩） |
| **seed mask 策略** | N/A | `workspace_and_not_robot`（如 2D 追踪时未 mask）或 `none`（如 2D 追踪时已 mask） |
| **robot mask（3D 阶段）** | N/A | frame 0 FK → 投影到图像 → 圆形膨胀 + 形态学闭运算 |
| **内参缩放** | N/A | depth 分辨率与 RGB 分辨率不同时，自动缩放 K 矩阵 |
| **输出格式** | N/A | `scene.h5`：per-clip `scene_flows`(float16), `scene_normals`(int8), `scene_colors`(uint8), `scene_visibility`(bool), `scene_depth_valid_mask`(bool) + `initial_rgb`(JPEG) + `initial_depth`(uint16 mm) |

---

## 十、分布式 & 大规模运行

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **并行机制** | `gnu parallel` 进程池，每 episode 一个进程 | `--rank / --world_size` 静态分片，无节点间通信 |
| **GPU 感知** | `nvidia-smi -L | wc -l` 动态检测 GPU 数量 | 无（由外部调度器分配 CUDA_VISIBLE_DEVICES） |
| **GCS 访问** | ❌ 需手动 `mount_gcs.sh` 挂载后访问 | ✅ 原生 `gs://` 路径，`gcs_utils.enforce_gcs_cache_policy` 管理缓存 |
| **缓存策略** | 无 | `POINTWORLD_CACHE_DIR` 共享缓存，不同 stage 复用 |
| **幂等性** | ⚠️ 用文件锁追加成功 episode 列表 | ✅ 跳过已存在的 `.h5` / `.json` 输出文件 |
| **负载均衡** | gnu parallel 自动 | `random.seed(42); random.shuffle(all_paths)` 保证各 rank 负载均匀 |
| **Slurm/GKE 支持** | ❌ 需要自己包装 job array | ✅ 直接映射：`--rank=$SLURM_ARRAY_TASK_ID --world_size=$SLURM_ARRAY_TASK_COUNT` |
| **最大可扩展性** | 单机 GPU 数量 | 任意节点数，数据并行无上限 |
| **数据规模假设** | 小批量（每次指定 --count 个 episode） | 全量 >10 TB，必须分布式才实际可行 |
| **多节点数据集成** | ❌ 不支持 | ✅ `convert_wds.py --rank/--world_size` 分布式写 WDS shards |

---

## 十一、输出格式 & 后处理

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **深度输出** | `raw_depth.npz`（uint16 mm）；`original_raw_depth.npz`（wrist 原始备份） | `{uuid}_depth.h5`（带 timestamps，uint16 mm） |
| **外参输出** | `extrinsics.npz`：`{cam}_base_extrinsic`(4×4) + `{cam}_extrinsics`(N×4×4) | `{uuid}_cameras.json`：含 vggt + optimized extrinsics + intrinsics |
| **视频输出** | ✅ `video_left/right.mp4` + `video_left/right_raw.mp4`（4 路视频存档） | ❌ 无视频文件输出 |
| **运动学输出** | `robot.npz`：joint_positions, gripper_positions, T_ee_base_all, T_cam_ee_init, valid_indices | 内嵌于 `.h5` 的 proprio 组 |
| **点流输出** | ❌ 无 | `scene.h5` per clip |
| **最终训练格式** | ❌ 无，需手动后处理 | WDS `.tar` shards（含 train/test 分割，`convert_wds.py` 生成） |
| **数据完整性检查** | ❌ 无 | `data_integrity_check.py`（验证每个 clip 的 key 完整性，生成 `integrity_check.json`） |
| **train/test 分割** | ❌ 无 | `make_wds_manifest.py`（seed=42，test=10%，与 paper split 可精确对齐） |

---

## 十二、可视化

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline |
|------|--------------------|-----------------------------|
| **可视化工具** | 无内置 | ✅ `visualization/visualize_generated_h5.py`（viser 3D viewer） |
| **可视化内容** | N/A | 3D 点流 + robot URDF + 场景点云，从 `.h5` 直接加载 |
| **随机浏览** | N/A | `--seed` 控制随机 H5/clip 选择，可复现 |
| **指定浏览** | N/A | `--h5_name` + `--clip_key`（如 `115:126`） |
| **robot 可视化** | N/A | URDF 采样 robot 点（`--max_robot_points`，默认 500，仅 gripper） |

---

## 十三、代码库依赖对比

### 你的 droid pipeline

| 类别 | 库 | 用途 |
|------|-----|------|
| Foundation Models | **VGGT-1B** (`facebook/VGGT-1B`) | 相机外参估计（HuggingFace 下载） |
| Foundation Models | **S2M2 XL** | 立体深度（local weights） |
| Foundation Models | **CoTracker3** | 加载但主流程未用（offline mode） |
| Foundation Models | **SAM ViT-H** | Gripper 分割（local weights） |
| 机器人 | **PyBullet** | OpenGL EGL 渲染机器人深度+分割 |
| 机器人 | **pybullet_data** | vanilla Panda URDF |
| 相机 | **pyzed / ZED SDK** | `.svo` 解码，内参提取 |
| 数学 | **PyTorch** | inference + Adam 优化 |
| 数学 | **scipy Rotation** | `R.from_euler("xyz", ...)` |
| 数学 | **numpy** | 点云变换、depth map |
| IO | **h5py** | 读 `trajectory.h5` |
| IO | **OpenCV** | `VideoWriter`，形态学（连通域/闭运算） |
| 系统 | **fcntl** | 多进程安全文件锁 |

### NVIDIA PointWorld pipeline

| 类别 | 库 | 用途 |
|------|-----|------|
| Foundation Models | **VGGT-1B** | 相机外参估计（本地 model.pt） |
| Foundation Models | **FoundationStereo** | 生产级深度预测（含 DINOv2 + Depth-Anything） |
| Foundation Models | **CoTracker3** | 稠密 2D 点追踪（核心步骤） |
| 机器人 | **urdfpy** | URDF 加载 + `visual_trimesh_fk` |
| 机器人 | **trimesh** | `mesh.sample(n)` 表面采样 |
| 几何 | **Open3D** | 法向量估计（KDTree + orient toward camera） |
| 空间 | **scipy ConvexHull** | Workspace boundary 凸包计算 |
| 图像 | **skimage.draw.polygon** | Workspace mask 多边形填充 |
| 数学 | **PyTorch** | inference + Adam/L-BFGS 优化 |
| 数学 | **transform_utils**（自研） | 四元数↔矩阵、`average_poses`（SO(3) 平均） |
| IO | **h5py** | 读写 depth/2d_flows/scene `.h5` |
| IO | **WebDataset** | WDS shards（训练格式） |
| IO | **gsutil / gcs_utils** | GCS 访问与缓存 |
| 可视化 | **viser** | 3D 交互式 viewer |
| 系统 | **random (seed=42)** | 可复现路径 shuffle |

---

## 十四、总结对比表

| 维度 | 你的 droid pipeline | NVIDIA PointWorld pipeline | 优势方 |
|------|--------------------|-----------------------------|--------|
| **任务完整性** | 外参标定为止 | 端到端训练数据生成 | NVIDIA ✅ |
| **Gripper 深度修复** | ✅ SAM + temporal median | ❌ 无 | 你 ✅ |
| **VGGT 初始化鲁棒性** | 第一帧（单帧） | 多帧 SO(3) 平均 | NVIDIA ✅ |
| **外参全局一致性** | ✅ Chamfer loss（3 视角联合） | ❌ 各相机独立 | 你 ✅ |
| **dual-base 竞争** | ✅ VGGT vs 预标定，择优 | ❌ 仅 VGGT | 你 ✅ |
| **Wrist 手眼标定** | ✅ 独立优化 `T_cam_ee` | ❌ 仅用运动学 GT | 你 ✅ |
| **机器人可见性预过滤** | ❌ 无 | ✅ 自动剔除无效帧 | NVIDIA ✅ |
| **参数数值条件** | ❌ 无归一化 | ✅ scale 归一化 | NVIDIA ✅ |
| **优化器选择** | Adam only | Adam 或 L-BFGS | NVIDIA ✅ |
| **渲染精度** | 像素级（OpenGL 光栅化） | 点云级（采样密度受限） | 你 ✅ |
| **渲染环境依赖** | 需要 EGL GPU 环境 | 纯 CPU，无依赖 | NVIDIA ✅ |
| **2D 点追踪** | ❌ 未集成 | ✅ CoTracker3 核心步骤 | NVIDIA ✅ |
| **3D 点流** | ❌ 无 | ✅ 完整 Flow2DTo3DConverter | NVIDIA ✅ |
| **法向量** | ❌ 无 | ✅ Open3D 估计 + int8 量化 | NVIDIA ✅ |
| **分布式支持** | 单机多卡 | 真分布式，任意节点数 | NVIDIA ✅ |
| **GCS 集成** | 手动挂载 | 原生 gcs_utils | NVIDIA ✅ |
| **幂等重跑** | 部分（文件锁记录） | 完整（跳过已有输出） | NVIDIA ✅ |
| **数据完整性** | ❌ 无 | ✅ integrity_check.py | NVIDIA ✅ |
| **train/test 分割** | ❌ 无 | ✅ manifest + paper split 对齐 | NVIDIA ✅ |
| **可视化工具** | ❌ 无 | ✅ viser 3D viewer | NVIDIA ✅ |
| **代码可维护性** | 单文件内联 | 模块化，类封装 | NVIDIA ✅ |
| **代码规模** | ~1200 行（简洁） | ~15000+ 行（完整） | 你 ✅（简洁） |
