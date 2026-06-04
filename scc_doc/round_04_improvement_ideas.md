# Round 04: 改进 idea 全景图 — 精度 / 速度 / CNN 迁移 / 闭环

生成时间：2026-06-01

---

## 0. 当前状态速览

| 指标 | 当前最优值 | 核心瓶颈 |
| --- | ---: | --- |
| Precision | 0.779 | ~22% false positive，主要来自背景干扰和虚假弧段组合 |
| Recall | 0.403 | **最大问题** — 近 60% 椭圆漏检 |
| FMeasure | 0.531 | 受 Recall 拖累严重 |
| AvgTime | ~15ms | 单图可接受，但批量/视频场景有优化空间 |

**Recall 为什么这么低？** 每个阶段都可能丢弃真正的椭圆：
- Canny 边缘断裂 → 椭圆边界不连续 → 弧段碎片化
- FSA 分割贪婪且阈值固定 → 弯弧被切碎或弱弧被丢弃
- 弧段分组贪心递归搜索 → 局部最优而非全局最优
- 验证阈值固定 `T_val=0.77` → 遮挡/模糊椭圆被拒绝
- 聚类贪心 → 部分正确检测被错误替代

[Round 03](round_03_transferable_ideas.md) 已经覆盖了 Top-K GeoScore、角度覆盖率、支持点重拟合、顶点感知分数、Soft-NMS、边缘引导验证。本文档聚焦**新 idea**，以及更深度的 CNN 架构和闭环设计。

---

## 1. 精度提升 idea（经典检测器）

### 1.1 弧段级别的改进

#### Idea A1: Multi-θ FSA — 多阈值弧段分割 + 重叠提案

**问题**：当前 FSA 使用全局固定的 `theta_fsa=PI/3` 和 `length_fsa=3.4`，且是贪婪非重叠分割。一个轮廓点只能属于一个弧段。如果真实椭圆在不同区域有不同的曲率特征，单一阈值要么切断弧段（丢 recall），要么合并噪声（降 precision）。

**方案**：
```
对同一条 dpContour 运行 3 组 (theta_fsa, length_fsa) 参数:
  - Relaxed: (PI/2.5, 5.0) — 宽松，产生长弧段
  - Default: (PI/3, 3.4)  — 当前设置
  - Strict:  (PI/4, 2.5) — 严格，产生短但干净的弧段
三组弧段一起进入 grouping → 互补覆盖
```

**CNN 迁移**：弧段提案网络（Arc Proposal Network）— 1D CNN/BiLSTM 沿轮廓序列，每个点输出 `P(start|point)`, `P(end|point)`，类似 temporal action detection。

**预期收益**：Recall +5~10%（更多弧段候选覆盖真实椭圆碎片），Precision 可能略降（需要更强的 grouping 来补偿）。

---

#### Idea A2: 弧段质量预过滤（Arc Quality Pre-Filter）

**问题**：当前弧段排序只按边缘点数（`SortArcs`），不评估"这个弧段看起来像椭圆的一部分吗？"大量非椭圆弧段进入昂贵的 grouping 阶段。

**方案**：
```
对每个 FSA 弧段计算快速特征:
  - curvature_consistency: 弧段内曲率方差（椭圆弧曲率变化应平滑）
  - chord_ratio: 弦长 / 弧长（太弯曲或太平直都不是好椭圆弧）
  - gradient_alignment: 弧段上各点的实际梯度方向与弧段法向的一致性
三个特征加权融合 → arc_quality_score
丢弃最低 20% 的弧段（不进入 grouping）
```

**CNN 迁移**：Arc Classifier — 小型 PointNet 或 1D CNN，输入弧段点序列，输出 `P(belongs_to_ellipse)`。

**预期收益**：Precision +3~5%，Speed +10~15%（减少无效 grouping 候选）。

---

#### Idea A3: 弧段嵌入学习（Arc Embedding for Grouping）

**问题**：当前弧段兼容性判断依赖手工规则（区域约束 + 曲率方向约束），需要大量调参，且容易在边缘 case 出错。

**方案（经典版）**：
```
对每对候选弧段 (i, j)，计算更丰富的 compatibility 特征:
  - 端点距离和方向
  - 曲率连续性和一致性
  - 弧段所在局部边缘密度
  - 两点间连线上是否有边缘支持（bridge evidence）
用逻辑回归/小型 MLP 替代硬阈值判断 → 输出 P(compatible)
```

**CNN 迁移（核心 idea）**：
构建**弧段图** `G = (V, E)`：
- 节点 V：每个弧段，特征为 (points, curvatures, mean_gradient, position)
- 边 E：候选弧段对
- 使用 **Graph Neural Network (GCN/GAT)** 学习节点嵌入，使同一椭圆弧段的嵌入相近
- 用 triplet loss 训练：同一椭圆弧段对 → 近；不同椭圆弧段对 → 远
- Grouping 变成 embedding space 中的聚类

这是**最自然的 CNN 迁移路径之一**，因为弧段分组天然是图问题。

**预期收益**：经典版 +3~5% FMeasure；GNN 版可望大幅提升 grouping 质量。

---

### 1.2 拟合级别的改进

#### Idea A4: RANSAC 增强椭圆拟合

**问题**：当前 DLS 拟合直接把所有弧段累积矩阵做特征分解。如果 grouping 混入了不属于该椭圆的弧段点（outlier arcs），拟合结果会被污染。

**方案**：
```
对每个候选组合:
  1. 从弧段点集中随机采样 3 个子集（每组足够拟合椭圆）
  2. 对每个子集拟合椭圆 → 计算所有弧段点对该椭圆的 support（距离 < threshold）
  3. 选择 support 最多的拟合
  4. 用所有 inlier 点重新 DLS 拟合
```

**CNN 迁移**：Deep Ellipse Fitting — PointNet 风格的网络，输入无序点集，直接输出椭圆参数。配合可微分拟合层实现端到端训练。

**预期收益**：Precision +3~5%（减少坏拟合导致的 false positive），IoU=0.8 下的精度提升更明显。

---

#### Idea A5: 加权最小二乘 + 迭代重加权（IRLS）

**问题**：DLS 拟合在存在 outlier 弧段时不稳定。当前所有弧段点等权。

**方案**：
```
1. 初始 DLS 拟合
2. 计算每个弧段点到拟合椭圆的距离
3. 用 Huber/Tukey 权重函数给每个点赋权（远离椭圆的点权重 → 0）
4. 用加权 DLS 重新拟合
5. 迭代 2-4 直到收敛（通常 3-5 轮）
```

**CNN 迁移**：可微分的 IRLS 层 — 每次迭代都是矩阵运算，可嵌入网络实现端到端梯度传播。

**预期收益**：稳健性提升，尤其对部分遮挡的椭圆。

---

### 1.3 验证级别的改进

#### Idea A6: 尺寸自适应验证阈值

**问题**：`T_val=0.77` 对所有尺寸的椭圆一视同仁。小椭圆的采样点少 → 分数方差大 → 应该降低阈值；大椭圆采样点多 → 分数更可靠 → 可以更严格。

**方案**：
```
T_val_adaptive = T_val_base + alpha * log(R + r) + beta * (R/r)
其中:
  - R+r 越大 → 椭圆周长越大 → 采样点越多 → 可略微收紧
  - R/r 越大 → 椭圆越扁 → 梯度对齐更困难 → 可略微放宽
alpha, beta 通过 grid search 在验证集上确定
```

**CNN 迁移**：验证网络直接接收椭圆参数 + 图像 patch → 输出置信度，自然学会尺寸相关的校准。

**预期收益**：Recall +2~4%（小椭圆和扁椭圆更不容易被误拒）。

---

#### Idea A7: 负证据验证（Negative Evidence Validation）

**问题**：当前验证只看"椭圆边界上有没有边缘支持"（正证据）。如果一个候选椭圆内部充满边缘（如纹理区域），或者外部有明显 contradicting 边缘，当前方法不会惩罚。

**方案**：
```
对每个候选椭圆:
  - inner_band: 椭圆内部收缩 10% 的环带 → 采样，期望 edge_hit 少
  - outer_band: 椭圆外部扩张 10% 的环带 → 采样，期望 edge_hit 也少（除非同心椭圆）
  - negative_score = (inner_edge_density + outer_edge_density) / boundary_edge_density
  - final_score = positive_score - w_neg * negative_score
```

**CNN 迁移**：Polar Transformer Validation — 将图像 patch 按椭圆参数 warp 到极坐标 → 椭圆边界变成水平线 → 用 2D CNN 同时检查边界支持（水平线有边缘）和内外抑制（上下区域无边缘）。

**预期收益**：Precision +3~5%（有效抑制复杂纹理背景的 false positive）。

---

#### Idea A8: 极坐标变换验证（Polar Validation CNN）

这是一个专门的 CNN 迁移 idea，同时也可以做经典版本。

**经典版本**：
```
对每个候选椭圆:
  1. 在极坐标下采样 M 个角度 × N 个径向距离 = M×N 的极坐标图像
  2. 椭圆边界在极坐标下是 θ 轴附近的一条带
  3. 统计该带内的边缘密度和梯度一致性
  4. 这比逐点采样更鲁棒，天然具有旋转不变性
```

**CNN 版本**：
```
PolarValidationNet:
  Input:  polar_image (M×N, 从候选椭圆参数 warp 得到)
  Backbone: 小型 2D CNN (如 ResNet-18 的前几层)
  Head:    FC → 1 (validation score)
  训练:    正样本 = GT 椭圆 warp；负样本 = 随机椭圆 warp
  特点:    旋转不变性天然满足，尺度归一化后也尺度不变
```

**预期收益**：CNN 版本可望成为最有效的验证模块，大幅超越手工验证分数。经典版本 +2~3% FMeasure。

---

### 1.4 后处理级别的改进

#### Idea A9: 全局能量最小化检测选择

**问题**：当前聚类/NMS 是贪心的 — 按分数排序，逐个决定保留或丢弃。这不是全局最优。

**方案**：
```
定义能量函数:
  E(S) = Σ_{i∈S} (1 - score_i) + λ Σ_{i,j∈S, i≠j} IoU_penalty(i, j)
其中 S 是被选中的检测集合。

用贪心+局部搜索近似求解（不需要精确最优）:
  1. 贪心初始化
  2. 随机 swap/remove/add 检测 → 若能量下降则接受
  3. 重复直到收敛
```

**CNN 迁移**：可学习的 NMS 模块 — 用网络预测 IoU，或直接用 Optimal Transport / Hungarian Matching 做可微分检测选择。

**预期收益**：Precision +1~3%（更好的去重和选择）。

---

## 2. 速度提升 idea

### 2.1 积分边缘图加速验证

**问题**：`fastValidation` 对每个候选椭圆做 VALIDATION_NUMBER 次逐点查询，每次查询遍历边缘链表找最近边缘点。这是当前最耗时的单步。

**方案**：
```
预计算积分边缘图 (Integral Edge Map):
  - IEM[x][y] = 以 (0,0) 到 (x,y) 为对角线的矩形区域内边缘像素数
  - 椭圆边界上任一弧段的边缘命中数 = O(1) 矩形查询
  - 梯度一致性也可以用积分梯度图加速

替代逐点链表遍历 → O(N) 变 O(1) per sample
```

**预期收益**：验证阶段加速 3~10×，总体加速 1.5~2×。

---

### 2.2 级联早退验证（Cascade Early-Reject）

**问题**：`fastValidation` 对所有候选做完整 360 点采样，即使这个候选在采样 30 个点后已经明显不合格。

**方案**：
```
Stage 1 (Coarse, ~30 samples): 快速检查大间隔采样 → 若分数 < 0.5*T_val → 直接拒绝
Stage 2 (Medium, ~90 samples): 中等采样 → 若分数 < 0.8*T_val → 拒绝
Stage 3 (Full, all samples):   完整采样，仅对通过前两阶段的候选
```

**预期收益**：验证加速 2~3× for most candidates（大部分候选在前两阶段被拒）。

---

### 2.3 弧段分组并行化

**问题**：`BiDirectionVerification` 对每个 root arc 是独立的。当前串行处理。

**方案**：
```
将 FSA_ArcContours 按空间位置分桶（如 4 象限或网格）
每个桶内的 root arc 独立处理 → OpenMP 并行
跨桶的边界弧段需要额外处理（少量）
```

**CNN 迁移**：弧段分组本身可完全并行化 — 图神经网络中所有节点同时更新。

**预期收益**：分组阶段加速 2~4×（多核），总体加速 1.3~1.8×。

---

## 3. CNN 迁移 — 核心架构设计

### 3.1 方案一：Hybrid Classic+CNN（渐进式，最推荐起步）

```
┌─────────────────────────────────────────────────────┐
│                  输入图像                             │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────┐           │
│  │  经典前端 (不动)                        │           │
│  │  Canny → contours → DP → FSA arcs     │           │
│  └──────────────────────────────────────┘           │
│                      │                               │
│                      ▼ (弧段列表)                     │
│  ┌──────────────────────────────────────┐           │
│  │  CNN 模块 1: Arc Grouper (GNN)        │           │
│  │  输入: 弧段图 G=(V,E)                  │           │
│  │  输出: 每个弧段的椭圆分配               │           │
│  └──────────────────────────────────────┘           │
│                      │                               │
│                      ▼ (弧段组)                       │
│  ┌──────────────────────────────────────┐           │
│  │  经典拟合 (不动)                        │           │
│  │  DLS ellipse fitting                  │           │
│  └──────────────────────────────────────┘           │
│                      │                               │
│                      ▼ (候选椭圆)                     │
│  ┌──────────────────────────────────────┐           │
│  │  CNN 模块 2: Polar Validator (2D CNN) │           │
│  │  输入: 极坐标 warp 图像 patch           │           │
│  │  输出: 校准后的置信度分数                │           │
│  └──────────────────────────────────────┘           │
│                      │                               │
│                      ▼                               │
│                  最终检测                             │
└─────────────────────────────────────────────────────┘
```

**优势**：
- 不需要海量标注数据（GNN 可用经典检测器生成伪标签训练；Validator 可用 GT 和随机负样本训练）
- 经典前端保留，鲁棒性有保底
- 两个 CNN 模块独立训练，可逐步替换
- 训练过程中可收集 failure case → 迭代改进

---

### 3.2 方案二：End-to-End Ellipse Detection CNN（完全替换）

参考 ElDet (ACCV 2022) 和 Ellipse R-CNN (arXiv 2020) 的设计：

```
┌────────────────────────────────────────────────────┐
│  Backbone: ResNet-50/101 + FPN                      │
│                      │                              │
│          ┌───────────┴───────────┐                  │
│          ▼                       ▼                  │
│  ┌──────────────┐       ┌──────────────┐           │
│  │ Edge Head    │       │ Ellipse Head │            │
│  │ (辅助任务)    │       │ (主任务)      │            │
│  │ 预测边缘图    │       │ 回归 5 参数   │            │
│  │ + 边缘法向    │       │ + confidence │            │
│  └──────────────┘       └──────────────┘           │
│          │                       │                  │
│          └───────────┬───────────┘                  │
│                      ▼                              │
│              ┌──────────────┐                       │
│              │ Vertex Head  │  (辅助任务)             │
│              │ 预测 4 顶点   │                       │
│              │ heatmap      │                       │
│              └──────────────┘                       │
│                      │                              │
│                      ▼                              │
│              ┌──────────────┐                       │
│              │ Coverage Head│  (辅助任务)             │
│              │ 角度覆盖率     │                       │
│              │ 预测           │                      │
│              └──────────────┘                       │
│                      │                              │
│                      ▼                              │
│            可微分椭圆 NMS + 输出                      │
└────────────────────────────────────────────────────┘
```

**关键设计选择**：

| 设计维度 | 推荐选项 | 理由 |
| --- | --- | --- |
| Backbone | ResNet-50 + FPN | 经典可靠，P3-P7 覆盖多尺度椭圆 |
| Anchor | Anchor-free (FCOS-style) | 椭圆尺寸变化大，anchor 设计困难 |
| 参数化 | (cx, cy, log(a), log(b), theta) | 正值约束 + 角度模 π 处理 |
| 损失函数 | Ellipse IoU Loss + L1 | IoU 比 L2 更适合评估指标 |
| 边缘辅助 | HED-style side outputs | 提供几何先验，加速收敛 |

---

### 3.3 方案三：可微分 AAMED（终极闭环）

将经典 AAMED 的关键步骤全部替换为可微分的对应模块：

```
┌──────────────────────────────────────────────────────┐
│  输入图像 (可微)                                       │
│      │                                                │
│      ▼                                                │
│  Differentiable Edge Detector                          │
│  (小型 CNN, 如 HED-lite 或简单的 Conv+BN+ReLU 堆叠)     │
│  输出: soft edge probability map                       │
│      │                                                │
│      ▼                                                │
│  Differentiable Arc Proposal                           │
│  方案 1: 在 edge map 上用 soft-argmax 追踪              │
│  方案 2: 用 learnable contour tracing (强化学习梯度)     │
│  方案 3: 越过 tracing，直接用 edge map 做 Hough-like    │
│          弧段检测 (可微分 Hough Transform)              │
│      │                                                │
│      ▼                                                │
│  GNN Arc Grouper (可微分)                               │
│  弧段 → 节点特征 → 消息传递 → 聚类 → 椭圆分配            │
│      │                                                │
│      ▼                                                │
│  Differentiable Ellipse Fitting                        │
│  方案 1: 隐式微分通过特征分解 (O(n³) but exact grad)     │
│  方案 2: Deep Fitting (PointNet → 5 params)            │
│      │                                                │
│      ▼                                                │
│  Polar Validator CNN (可微分)                           │
│  极坐标 warp → 2D CNN → confidence score               │
│      │                                                │
│      ▼                                                │
│  损失: Ellipse IoU + confidence BCE + regularization    │
│  ← 梯度反向传播到所有模块 ←                              │
└──────────────────────────────────────────────────────┘
```

**关键挑战**：
- 轮廓追踪的可微分实现最难 → 建议用 soft edge map + spatial argmax 替代
- 特征分解的梯度需要处理特征值重合的退化情况
- 训练初期可能不稳定 → 建议分阶段训练（先训练边缘检测，再加弧段模块，最后端到端）

**为什么这是终极闭环**：
- 损失信号从最终椭圆检测质量一直传到边缘检测器
- 边缘检测器学会产生"对椭圆检测有利"的边缘（而不是"像 GT edge map"的边缘）
- 整个 pipeline 自动学习最优参数，不需要手工调参
- 可以在目标数据集上自监督微调

---

## 4. 闭环（Closed Loop）设计

### 4.1 Render-and-Compare 自监督闭环

```
┌─────────────────────────────────────────────┐
│                                              │
│   未标注图像 ──→ CNN 椭圆检测器                │
│                      │                       │
│                      ▼                       │
│               检测到的椭圆参数                  │
│                      │                       │
│                      ▼                       │
│          可微分椭圆渲染器                      │
│          (Differentiable Renderer)           │
│          输出: 合成边缘图                       │
│                      │                       │
│                      ▼                       │
│          与真实 Canny 边缘图对比                │
│          L_render = MSE / Dice / ...          │
│                      │                       │
│                      ▼                       │
│     ← 梯度回传更新 CNN (不需要 GT 标注!) ←      │
│                                              │
└─────────────────────────────────────────────┘
```

**关键技术**：
- **可微分椭圆渲染器**：给定 (cx, cy, a, b, theta)，生成 anti-aliased 椭圆边缘 mask。椭圆参数方程对参数可微 → 渲染过程天然可微。
- **渲染损失**：`L = ||render(ellipse_params) - canny_edges||²`，只计算椭圆边界附近的像素。
- **多椭圆渲染**：对多个检测椭圆，用 max-pooling 或 alpha compositing 合成最终边缘图。

**优势**：不需要 GT 标注！可以在任意数量的无标注图像上训练。

---

### 4.2 Pseudo-Label 迭代闭环

```
┌───────────────────────────────────────────┐
│                                            │
│  Round 1:                                   │
│  经典检测器 (宽松参数)                         │
│      │                                      │
│      ▼                                      │
│  伪标签 (高 Recall, 低 Precision)              │
│      │                                      │
│      ▼                                      │
│  训练 CNN v1                                 │
│      │                                      │
│      ▼                                      │
│  Round 2:                                   │
│  CNN v1 检测 + 经典检测器验证                   │
│  (CNN 提出的候选 → 用经典验证判别真假)            │
│      │                                      │
│      ▼                                      │
│  更高质量的伪标签                               │
│      │                                      │
│      ▼                                      │
│  训练 CNN v2                                 │
│      │                                      │
│      ▼                                      │
│  Round 3+: 重复...                           │
│  CNN 越来越强 → 伪标签越来越好 → CNN 更强        │
│                                            │
└───────────────────────────────────────────┘
```

**为什么有效**：
- 经典检测器的强项（边缘检测、几何约束）和 CNN 的强项（纹理理解、全局上下文）互补
- 每一轮 CNN 都能发现经典检测器漏掉的椭圆（比如在弱边缘区域但纹理明确的椭圆）
- 经典检测器作为"验证器"为 CNN 提供可靠的正负样本

---

### 4.3 Active Learning 人机协同闭环

```
┌─────────────────────────────────────────────┐
│                                              │
│  已标注数据集 (Prasad 198 张)                   │
│      │                                       │
│      ▼                                       │
│  训练初始模型 M_0                              │
│      │                                       │
│      ▼                                       │
│  对未标注池中每张图像:                            │
│    - 经典检测器 + CNN 同时预测                   │
│    - 计算 disagreement score                  │
│    - 选择 disagreement 最大的 K 张图像           │
│      │                                       │
│      ▼                                       │
│  人工标注这 K 张图像                             │
│      │                                       │
│      ▼                                       │
│  加入训练集 → 训练 M_1                          │
│      │                                       │
│      ▼                                       │
│  重复直到性能饱和                                │
│                                              │
└─────────────────────────────────────────────┘
```

**Disagreement Score 定义**：
```
对于每张图像:
  1. 经典检测器 → 椭圆集合 C
  2. CNN → 椭圆集合 D
  3. 对每个 c ∈ C, 计算 max_iou(c, D)，若 < 0.5 → 经典独有，可能是 CNN 漏检
  4. 对每个 d ∈ D, 计算 max_iou(d, C)，若 < 0.5 → CNN 独有，可能是 false positive 或 经典漏检
  5. disagreement = |经典独有| + |CNN独有|
选择 disagreement 最大的图像 → 包含最多"一个模型确信、另一个模型不确信"的信息
```

---

## 5. 合成数据生成 Pipeline

CNN 方法的关键瓶颈是**标注数据少**（Prasad 仅 198 张图）。合成数据是关键。

```
┌──────────────────────────────────────────────────┐
│  Synthetic Ellipse Image Generator                │
│                                                   │
│  For each generated image:                        │
│    1. 随机背景 (从纹理数据集采样，或程序化噪声/gradient)  │
│    2. 随机放置 N 个椭圆 (N ~ Poisson(3))              │
│       - 参数: cx,cy 在图像内均匀                     │
│       - a ∈ [10, min(W,H)/3], b = a * ratio       │
│       - ratio ∈ [0.3, 1.0]                        │
│       - theta ∈ [0, π]                            │
│    3. 添加真实感效果:                                │
│       - 高斯模糊 (不同 sigma)                        │
│       - 加噪声 (高斯/泊松/椒盐)                       │
│       - 部分遮挡 (随机矩形/圆形遮挡物)                  │
│       - 光照变化 (渐变/阴影)                          │
│       - 纹理映射 (椭圆表面贴纹理)                      │
│    4. Domain randomization:                        │
│       - 背景类型随机                                  │
│       - 椭圆边缘类型: 实线/虚线/渐变/双线                │
│       - 椭圆内部: 填充/空心/半透明                     │
│    5. 自动生成 GT:                                  │
│       - 椭圆参数 (精确)                               │
│       - 椭圆 mask                                   │
│       - 椭圆边界采样点                               │
│       - 四个顶点坐标                                 │
│       - 可见性 mask (遮挡区域标记)                    │
│                                                   │
│  输出: (image, GT_ellipses, masks, vertices, ...)  │
└──────────────────────────────────────────────────┘
```

**关键原则**：
- 合成数据的分布要比真实数据更广（domain randomization）
- 先从纯合成数据预训练 → 再在 Prasad 上微调
- 合成时故意制造困难 case：严重遮挡、极扁椭圆（a/b > 5）、弱边缘、背景干扰

---

## 6. 多任务学习架构

一个统一的 CNN backbone，多个任务头共享特征：

```
┌────────────────────────────────────────────┐
│              Shared Backbone                │
│           (ResNet-50 + FPN)                 │
│                     │                       │
│     ┌───────┬───────┼───────┬───────┐      │
│     ▼       ▼       ▼       ▼       ▼      │
│  ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐     │
│  │Edge ││Ellip││Vert ││Occul││Comp │      │
│  │Head ││Head ││Head ││Head ││Head │      │
│  │     ││     ││     ││     ││     │      │
│  └─────┘└─────┘└─────┘└─────┘└─────┘     │
│  边缘   椭圆   顶点   遮挡   完整度            │
│  检测   检测   检测   预测   预测             │
└────────────────────────────────────────────┘

任务定义:
- Edge Head:    预测边缘图 + 边缘法向 (辅助，提供几何先验)
- Ellipse Head: 回归椭圆参数 + confidence (主任务)
- Vertex Head:  预测四个顶点 heatmap (辅助，改善轴精度)
- Occlusion Head: 预测遮挡 mask (辅助，帮助识别不完整椭圆)
- Completeness Head: 预测每个像素属于完整椭圆还是碎片 (辅助)
```

**为什么多任务有效**：
1. 边缘检测 → 强迫 backbone 学习几何特征
2. 顶点检测 → 轴参数的关键点监督
3. 遮挡预测 → 帮助模型理解"椭圆不完整是因为遮挡"vs"本来就不是椭圆"
4. 所有任务共享 backbone → 推理时只有 Ellipse Head 是必须的，其他 head 可丢弃

---

## 7. 面向 CNN 的特定技术模块

### 7.1 Ellipse IoU Loss

椭圆 IoU 计算困难（没有解析解）。实用方案：

**方案 A: 采样近似** — 沿两个椭圆边界各采样 N 个点，统计落在对方内部的点比例。
```
EllipseIoU ≈ (|A_points ∩ B_interior| + |B_points ∩ A_interior|) / 
             (|A_points| + |B_points|)
```

**方案 B: 旋转变换标准化** — 将两个椭圆通过仿射变换变为圆 → 圆-圆 IoU 有解析近似。
```
1. 计算将椭圆 A 变为单位圆的仿射变换 T_A
2. 将椭圆 B 也应用 T_A → 变为另一个椭圆 B'
3. 对单位圆和 B' 计算重叠面积（可以用离散积分近似）
```

**方案 C: 可微分渲染 IoU** — 用可微分渲染器将两个椭圆渲染到小 patch → 逐像素计算 IoU。

推荐方案 A（简单、可微、精度足够，N=360 即可）。

---

### 7.2 椭圆参数化的数值稳定性

直接回归 (cx, cy, a, b, theta) 有问题：
- a, b 必须为正
- theta 在 [0, π) 是周期的
- a 和 b 的 scale 可以差很多

**推荐参数化**：
```
网络输出: (cx', cy', log_a, log_b, sin_2θ, cos_2θ)
解码:
  cx = sigmoid(cx') * W       # 归一化到图像宽度
  cy = sigmoid(cy') * H       # 归一化到图像高度
  a = exp(log_a) * scale      # 正值保证
  b = exp(log_b) * scale      # 正值保证
  theta = 0.5 * atan2(sin_2θ, cos_2θ)  # 周期处理正确
```

优势：
- `log_a, log_b` 自然保证正值
- `(sin_2θ, cos_2θ)` 处理了 π 周期性（因为 sin(2(θ+π)) = sin(2θ)）
- 将回归目标做归一化 → 训练更稳定

---

### 7.3 椭圆 NMS 的可微分近似

标准 NMS 不可微（hard decision）。用于端到端训练的替代方案：

**Soft-NMS**：
```
score_i = score_i * Π_{j: score_j > score_i} exp(-IoU(i,j)² / σ)
```
可微，因为只是连续函数。

**Optimal Transport NMS**：
```
将检测选择形式化为 Optimal Transport 问题:
  - 供给方: 所有候选检测（带分数）
  - 需求方: K 个检测槽位
  - 运输成本: 检测分数 + IoU 惩罚
  - 用 Sinkhorn 算法迭代求解（可微！）
```

---

## 8. 优先级排序

### 按影响 × 可行性排序

| 优先级 | Idea | 类型 | 预期 FMeasure 提升 | 实现难度 | CNN 可迁移性 | 闭环贡献 |
|:---:| --- | --- |:---:|:---:|:---:|:---:|
| ★★★★★ | **A1: Multi-θ FSA** | Classic | +3~8% | 低 | 中 (Arc Proposal Net) | — |
| ★★★★★ | **A6: 尺寸自适应 T_val** | Classic | +2~4% | 极低 (改 3 行) | 中 | — |
| ★★★★★ | **A7: 负证据验证** | Classic | +3~5% | 低 | 高 (Polar Validator) | — |
| ★★★★☆ | **A4: RANSAC 拟合** | Classic | +3~5% | 中 | 中 (Deep Fitting) | — |
| ★★★★☆ | **A3: 弧段嵌入 + GNN** | Hybrid | +5~10% | 高 | **极高** | ★★★ |
| ★★★★☆ | **A8: 极坐标验证 CNN** | CNN | +5~10% | 中 | **极高** | ★★★ |
| ★★★★☆ | **A9: 能量最小化 NMS** | Classic | +1~3% | 中 | 中 | — |
| ★★★★☆ | **A5: IRLS 拟合** | Classic | +2~4% | 低 | 中 | — |
| ★★★☆☆ | **A2: 弧段质量预过滤** | Classic | +3~5% Prec | 低 | 中 (Arc Classifier) | — |
| ★★★☆☆ | **2.1: 积分边缘图** | Classic | 速度 1.5~2× | 中 | — | — |
| ★★★☆☆ | **2.2: 级联早退** | Classic | 速度 2~3× | 低 | — | — |
| ★★★☆☆ | **2.3: 并行分组** | Classic | 速度 2~4× | 中 | 高 | — |
| ★★★☆☆ | **4.1: Render-and-Compare** | CNN | +5~15% | 高 | **极高** | ★★★★★ |
| ★★★☆☆ | **4.2: Pseudo-Label 迭代** | Hybrid | +10~20% | 高 | **极高** | ★★★★★ |
| ★★☆☆☆ | **3.1: Hybrid Classic+CNN** | Hybrid | +10~20% | 很高 | **极高** | ★★★★ |
| ★★☆☆☆ | **3.2: End-to-End CNN** | CNN | +15~30%? | 很高 | **极高** | ★★★★ |
| ★★☆☆☆ | **3.3: 可微分 AAMED** | Hybrid | +10~20% | 极高 | **极高** | ★★★★★ |
| ★★☆☆☆ | **5: 合成数据 Pipeline** | 基础设施 | 间接 | 高 | **极高** | ★★★★★ |
| ★★☆☆☆ | **6: 多任务学习** | CNN | +5~10% | 高 | **极高** | ★★★ |
| ★★☆☆☆ | **4.3: Active Learning** | 基础设施 | 间接 | 中 | — | ★★★★★ |

---

## 9. 建议执行路线图

### Phase 1: 快速收益（1-2 天，纯 Classic）
```
1. A6: 尺寸自适应 T_val (30 分钟)
2. A7: 负证据验证 (2 小时)
3. A1: Multi-θ FSA (3 小时)
4. 在 Prasad 198 图上评估
目标: FMeasure +5~10%
```

### Phase 2: 稳健改进（3-5 天，Classic + 简单 ML）
```
5. A5: IRLS 拟合 (2 小时)
6. A4: RANSAC 拟合替代方案 (4 小时)
7. A2: 弧段质量预过滤 (3 小时)
8. A9: 能量最小化 NMS (4 小时)
目标: FMeasure 额外 +3~5%
```

### Phase 3: CNN 起步（1-2 周）
```
9. 5: 合成数据生成 Pipeline
10. A8: 极坐标验证 CNN (训练 + 集成)
11. 用经典检测器 + CNN 验证器跑 Prasad → 评估
目标: FMeasure 额外 +5~10%
```

### Phase 4: 深度 CNN 迁移（2-4 周）
```
12. A3: GNN 弧段分组
13. 3.1: Hybrid Classic+CNN 完整 pipeline
14. 在合成数据上预训练，Prasad 上微调
目标: FMeasure 接近或超越纯 CNN 方法
```

### Phase 5: 闭环（持续）
```
15. 4.2: Pseudo-Label 迭代闭环
16. 4.1: Render-and-Compare 自监督
17. 3.3: 可微分 AAMED 探索
目标: 持续自我改进，减少对标注数据的依赖
```

---

## 10. 关键参考论文（补充 Round 03 列表）

| 论文 | 关键 idea | 与本项目的关系 |
| --- | --- | --- |
| ElDet (ACCV 2022) | Anchor-free ellipse detection + edge guidance | 端到端 CNN 方案 |
| Ellipse R-CNN (arXiv 2020) | R-CNN + ellipse regression head | 二阶段方案参考 |
| Deep Hough Transform (ECCV 2020/PAMI) | 可微分 Hough 变换 | 可微分弧段检测 |
| PointNet (CVPR 2017) | 无序点集深度学习 | Deep Ellipse Fitting |
| Graph Attention Networks (ICLR 2018) | 注意力图神经网络 | 弧段 GNN 分组 |
| Spatial Transformer Networks (NeurIPS 2015) | 可微分空间变换 | 极坐标验证 warp |
| Soft-NMS (ICCV 2017) | 软非极大值抑制 | 可微分 NMS |
| Sinkhorn Distances (NeurIPS 2013) | 可微分最优传输 | Optimal Transport NMS |
| RCF: Richer Convolutional Features (CVPR 2017) | 边缘检测 CNN | 替代 Canny |
| Domain Randomization (IROS 2017) | 合成数据多样性 | 合成数据 Pipeline |
| Mean Teacher (NeurIPS 2017) | 自训练一致性 | Pseudo-Label 闭环 |
| Neural 3D Mesh Renderer (CVPR 2018) | 可微分渲染 | Render-and-Compare 闭环 |
| Deep Implicit Layers (NeurIPS 2020) | 隐式微分 | 可微分特征分解拟合 |

---

## 11. 与 Round 03 的关系

本文档是 Round 03 的延续和深化：

| Round 03 Idea | 本文档对应/深化 |
| --- | --- |
| Top-K + GeoScore 重排 | 保留，作为 A9 的基础 |
| Angular / Quadrant Coverage | 深化 → A8 极坐标验证 + Polar CNN |
| 支持点 Robust Refit | 深化 → A4 RANSAC + A5 IRLS |
| Vertex-aware 支持分数 | 深化 → Section 6 多任务 Vertex Head |
| Soft Ellipse-NMS | 深化 → Section 7.3 可微分 NMS |
| Edge-guided 旁路验证 | 深化 → A7 负证据 + Section 6 Edge Head |

**本文档新增的核心贡献**：
- Multi-θ FSA (A1) — 弧段级改进
- Arc Embedding + GNN (A3) — 图神经网络弧段分组
- 尺寸自适应验证阈值 (A6) — 最简单的快速收益
- Polar Validation CNN (A8) — 专用 CNN 验证架构
- 可微分 AAMED (Section 3.3) — 终极闭环
- Render-and-Compare 自监督 (Section 4.1) — 无需标注数据
- Pseudo-Label 迭代闭环 (Section 4.2) — 持续改进
- 合成数据 Pipeline (Section 5) — 训练数据基础设施
- 多任务学习架构 (Section 6) — CNN 架构设计
- 完整的 5 阶段执行路线图 (Section 9)
