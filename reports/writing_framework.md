# DROID Tech Report: 写作思路与大纲

> **目标**：把 DROID data generation pipeline 写成有 scientific contribution 的 tech report section，不只是描述做了什么，而要说清楚 **为什么这么做、和谁比、好多少**。

---

## 核心问题

写一篇好的 dataset pipeline 论文，需要回答三个层次的问题：

```
Why  → 为什么需要这个pipeline？先前方法的不足是什么？
How  → 具体怎么做的？每个design choice背后的reasoning是什么？
So What → 结果如何？和先前方法相比好多少？
```

---

## 1. 论文的 Scientific Narrative（故事线）

**核心故事**: DROID 数据集有丰富的多视角立体相机数据，但现有 pipeline（如 PointWorld）在提取可靠的 3D point tracks 时存在三个关键瓶颈：(1) depth quality on close-range surfaces, (2) multi-camera extrinsics accuracy, (3) cross-view tracking consistency。我们通过 **利用 DROID 特有的 stereo + robot kinematics 信号**，设计了一个 domain-aware pipeline 来解决这些问题。

### 故事线梗概

| 节拍 | 内容 | 目的 |
|------|------|------|
| Hook | DROID 是最大的真实世界机器人操作数据集之一，有多视角 stereo cameras，但要从中提取高质量 3D point tracks 极其困难 | 引出问题 |
| Gap | PointWorld 等先前工作使用 monocular depth + 固定 extrinsics + 单视角 tracking，这在 DROID 的 diverse setups 上不够好 | 定义差距 |
| Insight | DROID 的三个独特信号可以被利用：(1) calibrated stereo baselines, (2) synchronized robot kinematics, (3) multiple overlapping viewpoints | 核心洞察 |
| Method | 三阶段 pipeline：stereo depth + gripper refinement → differentiable extrinsics calibration → multi-view 3D fusion | 方案 |
| Evidence | 定量和定性结果表明显著改善 | 验证 |

---

## 2. 和 PointWorld 的详细对比

通过代码分析，以下是两个 pipeline 的关键差异：

### 2.1 Depth Estimation

| 维度 | PointWorld | Ours | Why Ours Is Better |
|------|-----------|------|-------------------|
| **Stereo model** | FoundationStereo | S2M2 (XL) | **待确认**: 需要实验结果。两者都是 stereo matching models，差异可能在 robustness 和 speed。S2M2 with `torch.compile` + `N_repeat=3` 可能更适合低纹理 robot 场景 |
| **Confidence filtering** | 无显式 confidence filtering（用 `us_right < 0` 做 validity check） | **Explicit confidence thresh = 0.95** | 更积极地过滤不可靠 depth，宁可丢掉 noisy pixels 也不引入误差 |
| **Gripper handling** | ❌ 不做特殊处理（gripper depth 和 environment 混在一起） | ✅ SAM mask + temporal median distillation + injection | **这是最大的差异之一**：wrist camera 的 gripper 是已知的刚体，但 stereo matching 在近距离光滑金属表面失效。我们利用 "gripper 在 closed 状态是静态的" 这一先验，用 temporal median 来 denoise |
| **Batch processing** | ✅ 支持 batch inference | ❌ Frame-by-frame | Our tradeoff: 简化代码 at cost of speed。可以提到 |

> **论文中怎么写 gripper depth refinement？**
> - **Problem**: "Stereo matching fails on the robot gripper due to specular metallic surfaces at close range (< 15cm), producing noisy or missing depth."
> - **Insight**: "The gripper is a rigid body; when closed, its geometry is constant across frames."
> - **Solution**: "We distill a clean depth surface via temporal median of masked stereo observations, then inject it back."
> - **Why not just use monocular depth?**: "Monocular depth provides relative depth only; stereo gives metric depth, which is essential for cross-view 3D consistency."

### 2.2 Extrinsics Calibration

| 维度 | PointWorld | Ours | Why Ours Is Better |
|------|-----------|------|-------------------|
| **Initial estimate** | VGGT (same) | VGGT (same) | 两者都用 VGGT 做初始化 |
| **Refinement strategy** | Differentiable alignment against robot depth (similar idea) | **Multi-stage**: per-camera independent → global joint Chamfer + Robot loss | **Global joint optimization** 是关键差异：不只是让每个相机对齐 robot，还要让多个相机之间的 environment point clouds 互相一致（Chamfer loss） |
| **Robot model** | URDF + urdfpy | URDF + PyBullet (EGL 光栅化) | 单 renderer：优化和 masking 用同一条路径。光栅化器天然只给可见表面，不需要像 CAD 采样那样用法线近似可见性——腕部相机上差别最大，夹爪自遮挡最重 |
| **Quality gate** | `filter_paths_by_extrinsics_quality.py` — 按 final loss 过滤，`max_final_loss=0.10` | No explicit quality gate in code（但可以在论文中讨论） | PointWorld 有一个 advantage：它显式过滤低质量 extrinsics scenes |
| **Multi-view consistency** | ❌ 每个 camera 独立优化 | ✅ Cross-camera Chamfer loss ensures consistency | 这对 3D tracking quality 至关重要 |

> **论文中怎么写 global joint optimization？**
> - **Problem**: "Independent per-camera calibration cannot guarantee multi-view geometric consistency — the same 3D point may project to inconsistent depths across views."
> - **Insight**: "Environment depth from multiple cameras, when correctly calibrated, should produce overlapping 3D point clouds."
> - **Solution**: "We add a truncated Chamfer distance loss between environment point clouds, jointly optimizing all cameras."
> - **Tradeoff**: "This is more expensive (3× more parameters) but dramatically improves cross-view consistency."

### 2.3 Tracking

| 维度 | PointWorld | Ours | Why Ours Is Better |
|------|-----------|------|-------------------|
| **Tracking model** | CoTracker3 (online, scaled_online.pth) | CoTracker3 (**offline** checkpoint) | Offline model sees the full video → more consistent long-range tracks |
| **Clip strategy** | Fixed-length clips (16 frames) with overlap → per-clip 2D tracks | **Full-episode** tracking (query_frame=0, grid_size=30) | 不需要 clip stitching 和 overlap handling，更简洁 |
| **Robot/env separation** | ❌ 不在 tracking 阶段区分 robot vs. environment | ✅ PyBullet robot mask → 分开处理 | Robot points 用 URDF FK 而不是 appearance-based tracking → 更准确 |
| **Robot tracking** | URDF FK for robot points (gripper only) | URDF FK for **all robot links** (arm + gripper) | More complete robot surface coverage |
| **Multi-view fusion** | ❌ Per-camera 2D → per-camera 3D，**不做 cross-view fusion** | ✅ Union-Find dedup → cross-view completion → median 3D fusion | **这是最大的差异**：PointWorld 每个相机独立产生 3D flows，我们的 pipeline 跨视角融合产生一致的 3D tracks |
| **Cross-view completion** | ❌ | ✅ 把一个视角发现的 3D point re-project 到其他视角，用 CoTracker query mode 补全 | 大幅增加 point coverage 和 visibility |
| **Quality filtering** | EE motion filtering（按 end-effector 运动量过滤 idle clips） | Min views + min frames filter | 不同的 quality philosophy |

> **最核心的 scientific contribution: Multi-View 3D Fusion**
>
> PointWorld 的设计目标是为每个单视角生成 3D flows（用于 single-camera world model training）。我们的目标不同：我们要生成 **multi-view consistent 3D point tracks**，即同一组 3D points 在所有视角中有一致的 2D projections 和 visibility。这需要 cross-view deduplication、completion 和 median fusion —— 这些是 PointWorld 完全不做的。

---

## 3. 需要补充的实验/证据

> **目前缺失**: 你的 codebase 里没有看到 evaluation metrics 或 quantitative comparison。要让论文有说服力，需要至少以下几类证据。

### 3.1 必须有的定量评估

| 评估类型 | 具体做什么 | Metric |
|---------|-----------|--------|
| **3D reprojection error** | 把 fused 3D tracks project 回每个 camera，和 2D tracks 比较 | Mean/median reprojection error (px) |
| **Multi-view consistency** | 对比 per-camera independently lifted 3D vs. fused 3D | 3D position variance across views |
| **Depth quality** | 比较 raw stereo depth vs. gripper-refined depth | RMSE on gripper region (if ground truth available)，或 temporal stability |
| **Extrinsics quality** | 比较 VGGT-only vs. Stage 2 refined vs. Stage 3 joint | Alignment error / Chamfer distance |
| **Ablation: gripper refinement** | With/without gripper depth injection | Tracking accuracy on wrist camera |
| **Ablation: cross-view fusion** | Single-view 3D vs. multi-view fused 3D | 3D stability / consistency |

### 3.2 可以有的定性可视化

| 可视化 | 说明 |
|--------|------|
| **Pipeline overview figure** | 3-stage flow with example intermediate outputs |
| **Gripper depth before/after** | Raw stereo → refined depth (side by side) |
| **Extrinsics alignment** | Robot CAD overlay on depth map before/after calibration |
| **Cross-camera point cloud** | Multiple cameras' environment point clouds in world frame |
| **2D tracking overlay** | CoTracker tracks on all views, with cross-view completed tracks highlighted |
| **3D track point cloud** | Final 3D trajectories colored by time/type |

### 3.3 和 PointWorld 的直接对比（如果可行）

| 对比 | 具体怎么做 |
|------|-----------|
| **在相同 episode 上跑两个 pipeline** | 选 100 个 episodes，分别跑 PointWorld 和 ours |
| **计算 3D track quality** | Reprojection consistency, temporal smoothness, coverage density |
| **计算 depth quality** | Per-pixel depth comparison (if both use stereo) |

---

## 4. 推荐的论文结构

```
Section X: DROID Data Generation Pipeline

X.1 Problem Setting and Motivation
    - DROID dataset 特点（multi-view stereo, robot kinematics）
    - 从 DROID 提取 3D point tracks 的挑战
    - 和 PointWorld 等先前工作的关系和不足

X.2 Pipeline Overview
    - 3 stage architecture 描述
    - Figure: pipeline overview with example outputs

X.3 Stage 1: Metric Depth with Gripper Refinement
    X.3.1 Stereo Depth via S2M2
        - Why stereo > monocular for metric tracks
        - Confidence thresholding rationale
    X.3.2 Gripper Depth Distillation
        - Problem: stereo fails on close-range specular surfaces
        - Insight: closed-gripper rigidity prior
        - SAM consensus mask + temporal median
        - Why not: model-based depth (noisy registration) / learned inpainting (no training data)

X.4 Stage 2: Differentiable Extrinsics Calibration
    X.4.1 Visual Initialization via VGGT
    X.4.2 Camera-Robot Depth Alignment
        - 6-DOF differentiable parameterization
        - Front-face culling and depth tolerance
    X.4.3 Global Joint Optimization
        - Why: independent calibration ≠ multi-view consistency
        - Chamfer + Robot loss formulation
        - Ablation: independent vs. joint

X.5 Stage 3: Multi-View 3D Point Tracking
    X.5.1 Per-View Dense 2D Tracking
        - CoTracker offline vs online choice
    X.5.2 3D Lift with Robot-Environment Separation
        - PyBullet segmentation for clean separation
        - Why: appearance-based tracking fails on robot (self-similar, reflective)
    X.5.3 Cross-View Deduplication
        - Union-Find 3D nearest-neighbor
        - Why 1.5cm radius
    X.5.4 Cross-View Completion
        - Multi-keyframe re-tracking
        - Why: dramatically increases per-view coverage
    X.5.5 Median 3D Fusion
        - Why median > mean (robustness to outliers)
        - Temporal smoothing

X.6 Robot Surface Tracking via Forward Kinematics
    - Why: URDF FK > appearance tracking for articulated robot
    - Link binding and FK propagation
    - Visibility model (self-occlusion + environment occlusion)

X.7 Experiments
    X.7.1 Dataset Statistics
    X.7.2 Ablation Studies
    X.7.3 Comparison with PointWorld (if available)
    X.7.4 Qualitative Results
```

---

## 5. 每个 Design Choice 的 Reasoning 清单

> 论文中每个 design choice 都要有 **"为什么这么做/为什么不用别的方法"** 的解释。以下是完整清单。

### Stage 1: Depth

| Choice | Why This | Why Not Alternatives |
|--------|----------|---------------------|
| **Stereo depth (S2M2) 而非 monocular depth** | Metric depth → 跨视角 3D 一致性。Scale-consistent across cameras | Monocular (e.g., DPT, Depth Anything) gives relative depth only; needs per-frame scale alignment which introduces error |
| **Confidence thresh = 0.95** | Aggressive filtering: prefer missing data over wrong data. Downstream stages (tracking, fusion) can handle sparse depth, but wrong depth propagates errors | Lower thresh → more coverage but noisy; higher thresh → too sparse |
| **SAM ViT-H for gripper mask** | Zero-shot; no need for robot-specific training data. Works across all DROID setups with different grippers/backgrounds | Fine-tuned segmentation model would need labeled data per setup; URDF projection mask has registration noise |
| **Consensus mask (voting) 而非 single-frame mask** | Single SAM prediction varies across frames; consensus stabilizes | Tracking-based mask propagation could work but adds complexity |
| **Temporal median for gripper depth** | Robust to per-frame stereo noise; produces crisp edges | Mean would blur; single-frame would keep noise |
| **只对 closed-gripper frames 做 distillation** | Gripper geometry 在 closed 时是确定性的；open 时手指位置变化 | Could extend to open gripper with FK-predicted finger positions (future work) |

### Stage 2: Extrinsics

| Choice | Why This | Why Not Alternatives |
|--------|----------|---------------------|
| **VGGT 初始化而非 PnP / COLMAP** | VGGT 从 2-3 张图直接出 pose，无需 feature matching / point correspondence | PnP 需要已知 3D-2D correspondences；COLMAP 需要 dense matching，在 robot scenes 里不稳定 |
| **Differentiable depth alignment 而非 ICP** | Depth re-projection loss 是 smooth 的，适合 gradient descent；ICP 有 local minima 问题 | ICP (point-to-point / point-to-plane) could work but needs good initialization and handles outliers poorly |
| **6-DOF axis-angle + translation** | 最小参数化，无需 manifold optimization | Quaternion: 4 params + normalization constraint; rotation matrix: 9 params + orthogonality constraint |
| **Global joint Chamfer loss** | 让多相机的 environment point clouds 在 world frame 中对齐 | Without this, each camera could have self-consistent but mutually inconsistent extrinsics |
| **Truncated Chamfer (5cm cutoff)** | Ignores outliers / non-overlapping regions | Full Chamfer dominated by large errors from non-overlapping regions |

### Stage 3: Tracking

| Choice | Why This | Why Not Alternatives |
|--------|----------|---------------------|
| **CoTracker offline 而非 online** | Offline sees full video → better long-range consistency | Online: faster, streaming-capable, but temporal drift |
| **Full-episode tracking 而非 clip-based** | No clip stitching artifacts; simpler code | PointWorld uses 16-frame clips → needs overlap stitching, loses long-range correspondences |
| **Union-Find dedup (1.5cm radius)** | Merges duplicate 3D points from different views without pairwise assignment | Hungarian matching: O(N³), doesn't scale; KD-tree + greedy: order-dependent |
| **Cross-view completion via multi-keyframe queries** | 5 keyframes ≫ 1 keyframe: handles occlusion at different moments | Single-keyframe: misses occluded points; all-frame: too expensive |
| **Median 3D fusion 而非 mean** | Robust to single-view outliers (wrong depth, bad extrinsics) | Mean: sensitive to outliers; weighted mean: needs weight estimation |
| **URDF FK for robot tracking** | Exact kinematic model → perfect 3D trajectories | Appearance-based tracking: robot is self-similar, reflective → tracking fails |
| **Gaussian temporal smoothing (σ=1.5)** | Removes sub-pixel jitter from quantization | No smoothing: noisy; larger σ: loses real motion |

---

## 6. 你需要决定的 Open Questions

以下问题需要你来回答，我无法从代码中推断：

1. **S2M2 vs. FoundationStereo**: 有没有做过 head-to-head comparison？哪个在 DROID 上更好？
2. **Episode 覆盖率**: 最终有多少 DROID episodes 成功跑完 3 stages？失败率是多少？
3. **3D track 统计**: 平均每个 episode 有多少 3D points？Average track length？
4. **定量评估**: 有没有跑过 reprojection error、Chamfer distance 等 metrics？
5. **和 PointWorld 的直接对比**: 有没有在相同 episodes 上跑过两个 pipeline 做对比？
6. **Downstream task 验证**: 这些 3D tracks 最终用在什么 downstream task 上（TAP training/eval）？有没有 downstream performance 数据？
7. **Compute cost**: 处理一个 episode 的三个 stage 各需要多久?

---

## 7. Action Items（按优先级）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 🔴 P0 | **准备定量实验数据** | 没有 numbers 就没有 paper。至少需要 reprojection error 和 ablation |
| 🔴 P0 | **回答 Open Questions** | 上述 7 个问题的答案决定论文的 framing |
| 🟡 P1 | **根据你的回答重写 draft** | 我来写，但 narrative 取决于你有什么数据 |
| 🟡 P1 | **准备 figures** | Pipeline overview, depth comparison, tracking visualization |
| 🟢 P2 | **和 PointWorld 做 head-to-head** | If time permits, this would make the paper much stronger |
| 🟢 P2 | **Downstream task eval** | 如果有 TAP benchmark results，加上去 |
