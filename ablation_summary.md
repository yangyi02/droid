# Ablation Study Summary

> [!NOTE]
> All metrics are averaged over 30 episodes. Lower is better for Chamfer, Robot depth, CoTrk px, and TAP px. Higher is better for BG%.

---

## Baseline

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| **E0** | **Baseline: yourdfpy, Chamfer+Robot** | **0.1261** | **0.0587** | **43.7%** | **38.4** | **36.3** | **194s** |

---

## 1. Rendering Backend

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E0 | Baseline (yourdfpy) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E1 | PyBullet backend | 0.1299 ❌ | 0.0631 ❌ | 42.9% | 42.3 ❌ | 34.5 ✅ | 699s ❌ |

> [!IMPORTANT]
> **结论**: PyBullet 全面劣于 yourdfpy，且速度慢 3.6x。yourdfpy 是更优选择。

---

## 2. Loss 组合消融

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E0 | Chamfer + Robot (baseline) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E2 | Robot only (no Chamfer) | 0.1372 ❌ | 0.0591 | 42.6% | 46.7 ❌ | 32.1 ✅ | 190s |
| E3 | Chamfer only (no Robot) | 0.1238 ✅ | 0.0813 ❌❌ | 44.7% | 114.4 ❌❌ | 88.2 ❌❌ | 189s |

> [!IMPORTANT]
> **结论**:
> - 去掉 Chamfer → Chamfer 指标恶化，tracking 略差
> - 去掉 Robot depth → Chamfer 略好，但 Robot 严重恶化 (+38%)，tracking 崩溃 (CoTrk 114.4, TAP 88.2)
> - **两个 loss 缺一不可**，Robot depth 对 tracking 质量尤其关键

---

## 3. Tracking Loss 消融

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E0 | Baseline (no tracks) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E4 | +CoTracker, w=0.001 | 0.1265 | 0.0590 | 43.7% | 32.8 ✅ | 40.4 | 209s |
| E5 | +TAPNext, w=0.001 | 0.1264 | 0.0594 | 43.8% | 44.1 | 32.5 ✅ | 225s |
| E6 | CoTracker only, w=0.0001 | 0.1261 | 0.0588 | 43.8% | 37.3 | 36.6 | 208s |
| E7 | CoTracker only, w=0.01 | 0.1317 ❌ | 0.0607 ❌ | 42.7% | 45.4 ❌ | 48.7 ❌ | 208s |
| E8 | CoTracker, grid=15 | 0.1270 | 0.0589 | 43.6% | 37.6 | 31.1 ✅ | 200s |
| E9 | CoTracker, grid=50 | 0.1296 ❌ | 0.0604 ❌ | 42.2% | 47.8 ❌ | 41.7 ❌ | 103s |
| E12 | PyBullet + CoTracker, w=0.001 | 0.1280 ❌ | 0.0593 | 42.6% | 63.7 ❌ | 39.4 | 836s |

> [!IMPORTANT]
> **结论**:
> - Tracking loss **权重很敏感**: w=0.01 过大导致全面恶化, w=0.0001 基本无影响, w=0.001 较为平衡
> - CoTracker (E4) 改善 CoTrk 但略伤 TAP; TAPNext (E5) 改善 TAP 但略伤 CoTrk — **各自对自己的 tracker 有偏好**
> - Grid=15 (sparse) 效果较好; Grid=50 (dense) 反而恶化
> - **总体来看 tracking loss 收益有限，权重需精细调节**

---

## 4. 学习率消融

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E0 | lr=0.001 (baseline) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E13 | lr=0.0003 (3x lower) | 0.1271 | 0.0589 | 43.7% | 34.4 ✅ | 28.6 ✅ | 188s |
| E14 | lr=0.003 (3x higher) | 0.1261 | 0.0592 | 43.7% | 41.0 | 30.4 ✅ | 188s |

> [!NOTE]
> **结论**: 学习率 3x 范围内影响不大。低 lr 在 tracking 上略有改善，可能优化更稳定。

---

## 5. 训练步数消融

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E15 | n_steps=200 (short) | 0.1263 | 0.0590 | 43.7% | 35.8 | 35.2 | 129s |
| E0 | n_steps=500 (baseline) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E16 | n_steps=1000 (long) | 0.1260 | 0.0591 | 43.8% | 38.0 | 34.7 | 285s |

> [!NOTE]
> **结论**: 200 步已基本收敛，增加到 1000 步收益很小。**200-500 步是最佳性价比区间**。

---

## 6. Robot Weight 消融

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E19 | robot_weight=0.1 (weak) | 0.1240 ✅ | 0.0646 ❌ | 44.6% | 54.2 ❌ | 45.2 ❌ | 188s |
| E0 | robot_weight=1.0 (baseline) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E20 | robot_weight=10.0 (strong) | 0.1337 ❌❌ | 0.0589 | 43.1% | 44.2 ❌ | 31.3 ✅ | 187s |

> [!IMPORTANT]
> **结论**: 
> - Weight 过低 → Chamfer 好但 Robot/tracking 差（类似 E3 的退化模式）
> - Weight 过高 → Chamfer 严重恶化
> - **robot_weight=1.0 是合理平衡点**

---

## 7. Chamfer 采样密度

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E21 | chamfer_n_points=1000 | 0.1263 | 0.0589 | 43.8% | 38.8 | 35.8 | 131s |
| E0 | chamfer_n_points=2000 (baseline) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E22 | chamfer_n_points=4000 | 0.1277 ❌ | 0.0623 ❌ | 45.8% ✅ | 27.6 ✅ | 26.1 ✅ | 61s |

> [!TIP]
> **结论**: 1000 和 2000 差异不大。4000 的 tracking 大幅改善但 Chamfer/Robot 恶化，且速度最快 (61s)。**如果只看 tracking 指标，chamfer_n_points=4000 很有吸引力**。

---

## 8. 评估 Grid 密度

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E23 | grid_size=15 (sparse eval) | 0.1265 | 0.0590 | 43.7% | 26.1 | 28.6 | 185s |
| E0 | grid_size=30 (baseline eval) | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E24 | grid_size=50 (dense eval) | 0.1260 | 0.0586 | 43.8% | 48.6 | 32.2 | 184s |

> [!NOTE]
> **结论**: Grid size 主要影响 tracking 评估的绝对数值，不影响 Chamfer/Robot。Sparse grid tracking error 更低（采样偏差），**建议固定 grid_size=30 作为标准评估**。

---

## 9. Multi-restart & Fine-tune

| ID | Description | Chamfer | Robot | BG% | CoTrk px | TAP px | Time |
|----|-------------|---------|-------|-----|----------|--------|------|
| E0 | Baseline | 0.1261 | 0.0587 | 43.7% | 38.4 | 36.3 | 194s |
| E10 | Stage 2 multi-restart | 0.1260 | 0.0585 ✅ | 43.8% | 32.9 ✅ | 35.3 | 450s |
| E17 | Baseline + Stage 4 fine-tune | 0.1243 ✅ | 0.0633 ❌ | 44.5% | 55.7 ❌ | 43.6 ❌ |  311s |
| E18 | CoTracker + Stage 4 fine-tune | 0.1253 ✅ | 0.0642 ❌ | 44.4% | 49.4 ❌ | 49.1 ❌ | 337s |

> [!WARNING]
> **结论**: 
> - Multi-restart (E10) 全面小幅改善，但 2.3x 时间开销
> - Stage 4 fine-tune 改善 Chamfer 但 **显著恶化 Robot 和 tracking** — 可能 overfit 到外观而忽略几何
> - **Stage 4 fine-tune 目前不推荐使用**

---

## 🏆 Key Takeaways

| 发现 | 推荐 |
|------|------|
| Rendering backend | ✅ yourdfpy（速度和质量均优于 PyBullet）|
| Loss 组合 | ✅ Chamfer + Robot 缺一不可 |
| Tracking loss | ⚠️ 收益有限，需精细调权重 |
| 学习率 | ✅ lr=0.001 或稍低均可 |
| 训练步数 | ✅ 200-500 步最佳性价比 |
| Robot weight | ✅ 1.0 是合理平衡点 |
| Chamfer 采样 | ✅ 2000 足够，4000 有 tracking 优势 |
| Multi-restart | ⚠️ 有效但慢 2.3x |
| Stage 4 fine-tune | ❌ 不推荐（恶化 tracking）|

> [!TIP]
> **最优配置建议**: E0 baseline (yourdfpy + Chamfer + Robot, lr=0.001, n_steps=200-500, robot_weight=1.0, chamfer_n_points=2000) 已经是非常强的配置。如果追求更好的 tracking，可以考虑 E13 (低 lr) 或 E22 (dense chamfer sampling)。
