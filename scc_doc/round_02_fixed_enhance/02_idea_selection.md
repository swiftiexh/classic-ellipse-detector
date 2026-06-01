# 候选 Idea 选择报告 — deepseek 分支

生成时间：2026-05-30
分支：deepseek（父分支 scc，已继承 isCombValid 修复）

## 1. 约束条件与背景

- 尽量不修改 `src/` 下 C++ 核心文件
- 优先新增 Python 脚本 / wrapper
- 低实现成本，方便合并
- CNN 可迁移
- 本分支已继承 scc 分支的 `src/Group.cpp` isCombValid 修复（FMeasure +0.007）
- 新的 idea 应在修复后的 detector 基础上寻找进一步改进空间

## 2. 候选 Idea

### Idea A: SCC-Enhance — 自适应图像增强预处理

**改进位置**: 检测流程最前端（图像输入 → 增强 → detector）

**输入**: 原始 BGR 图像
**输出**: 增强后的图像（写入临时目录，供 detector 读取）

**方法**: 对每张输入图像依次应用：
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) — 提升局部对比度，使暗区椭圆边界更可见
2. Bilateral Filter — 保边去噪，平滑均匀区域的同时保留锐利边缘

**动机**: baseline Recall 仅 0.40（修复后），说明大量椭圆仍被漏检。低对比度/暗区域的椭圆边界容易被 Adaptive Canny 漏掉。预处理增强边缘可见性后，可能提取到更多有效弧段。

**实现**: ~60 行 Python。增强图像 → 写入临时目录 → 调用 aamed_demo.exe → 收集 .fled.txt。

**是否修改 C++**: 否。

**CNN 迁移性**:

| 维度 | 分数 | 理由 |
|---|---|---|
| 输入输出清晰 | 2 | Image → Enhanced Image |
| 可微分潜力 | 2 | 可改造为可微分增强模块（如 learnable filter） |
| CNN 接口友好 | 2 | 可作为 preprocessing CNN block |
| 监督信号 | 1 | 间接监督，增强质量通过检测效果反馈 |
| batch 化潜力 | 2 | 图像级操作，完全可 batch |
| 几何意义 | 1 | 通用图像增强，非椭圆专属几何 |
| 工程独立性 | 2 | 完全独立模块 |
| **总分** | **12/14** | |

### Idea B: SCC-SoftScore — 边界支持率软评分重排序

**改进位置**: 检测流程末端（detector 输出 → 边界评分 → 重排序）

**输入**: .fled.txt 候选椭圆 + 原始图像
**输出**: 按新 confidence 重排序的椭圆列表

**方法**:
1. 对每张图计算 Canny edge map
2. 对每个检测椭圆，沿边界等距采样 N 点
3. 计算 boundary_edge_ratio = 采样点命中边缘的比例
4. 将 boundary_edge_ratio 与原始 fastValidation score 融合
5. 按新 confidence 重排序（soft re-ranking，不硬删除）

**与 GeoFilter 的关键区别**: GeoFilter 是 hard binary filter（几何规则 → 删/留），SoftScore 是 soft re-ranking（连续分数 → 重排序），信息损失更小。

**实现**: ~120 行 Python。读 .fled.txt → 加载图像 → 计算新分数 → 写入 .fled.txt。

**是否修改 C++**: 否。

**CNN 迁移性**:

| 维度 | 分数 | 理由 |
|---|---|---|
| 输入输出清晰 | 2 | Candidates + Edge Map → Confidence Scores |
| 可微分潜力 | 2 | 可改造为 confidence head、geometry loss 或 IoU predictor |
| CNN 接口友好 | 2 | 可作为 proposal head 的 confidence branch |
| 监督信号 | 2 | GT ellipse 可通过 IoU 自动生成正负标签 |
| batch 化潜力 | 1 | 候选数可变，需 padding/top-k |
| 几何意义 | 2 | 显式利用椭圆边界几何先验 |
| 工程独立性 | 2 | 完全独立模块 |
| **总分** | **13/14** | |

### Idea C: SCC-EdgeDensity — 边缘密度自适应参数选择

**改进位置**: 检测流程前端（分析图像边缘密度 → 选择 T_val 参数 → detector）

**输入**: 原始图像
**输出**: 每张图自适应检测参数

**方法**:
1. 计算 Canny edge density（边缘像素占比）
2. 将图像按密度分为 low / medium / high 三档
3. 不同档位使用不同 T_val（validation 硬阈值）:
   - Low density（少边缘）: T_val = 0.65（宽松）
   - Medium density: T_val = 0.77（默认）
   - High density（多边缘）: T_val = 0.85（严格）

**动机**: 单一阈值无法适应所有场景。低纹理图像（如光滑椭圆物体）需要宽松阈值检出更多候选；高纹理图像需要严格阈值抑制噪声误检。

**实现成本**: 中。需要编译 3 个不同 T_val 的 detector 二进制（或修改 main.cpp 让 T_val 可配置）。

**是否修改 C++**: 需要小幅修改（让 T_val 从命令行读取），或用 3 个预编译二进制。

**CNN 迁移性**:

| 维度 | 分数 | 理由 |
|---|---|---|
| 输入输出清晰 | 2 | Image → T_val Selection |
| 可微分潜力 | 1 | 可改造为 adaptive gating，但阈值选择本身不可微 |
| CNN 接口友好 | 2 | 可作为动态参数预测分支 |
| 监督信号 | 1 | 需要启发式标签或 RL |
| batch 化潜力 | 2 | 图像级操作 |
| 几何意义 | 1 | 基于图像统计而非椭圆几何 |
| 工程独立性 | 1 | 需要修改或绕开硬编码参数 |
| **总分** | **10/14** | |

## 3. 综合打分与排名

| 维度 (权重) | Idea A (Enhance) | Idea B (SoftScore) | Idea C (EdgeDensity) |
|---|---|---|---|
| 实现成本低 (25%) | 9 | 8 | 4 |
| 可合并性强 (20%) | 10 | 10 | 6 |
| CNN 迁移价值 (20%) | 8 | 9 | 7 |
| 精度提升潜力 (15%) | 7 | 3 | 6 |
| 速度提升潜力 (10%) | 3 | 5 | 4 |
| 实验风险低 (10%) | 9 | 10 | 5 |
| **加权总分** | **8.10** | **7.75** | **5.40** |

计算详情:
- A: 0.25×9 + 0.20×10 + 0.20×8 + 0.15×7 + 0.10×3 + 0.10×9 = 8.10
- B: 0.25×8 + 0.20×10 + 0.20×9 + 0.15×3 + 0.10×5 + 0.10×10 = 7.75
- C: 0.25×4 + 0.20×6 + 0.20×7 + 0.15×6 + 0.10×4 + 0.10×5 = 5.40

## 4. 排名表

| 排名 | Idea | 改进位置 | 预期收益 | 实现成本 | CNN 迁移分 | 风险 | 推荐 |
|---|---|---|---|---|---|---|---|
| 1 | SCC-Enhance | 预处理 | Recall ↑ | 低 | 12/14 | 低 | **是** |
| 2 | SCC-SoftScore | 后处理 | Precision ↑ | 低 | 13/14 | 低 | 备选 |
| 3 | SCC-EdgeDensity | 参数选择 | Recall+Precision | 中 | 10/14 | 中 | 否 |

## 5. 最终选择

选择 **Idea A: SCC-Enhance（自适应图像增强预处理）**。

核心理由：
1. **从源头改善** — 提升输入质量，而非在输出端筛选。GeoFilter 已证明后处理上限极低（+0.0004 FMeasure），因为 baseline NMS 已做了充分的候选筛选。
2. **Recall 提升潜力大** — baseline Recall 仅 0.40，说明 60% 的 GT 椭圆被漏检。增强低对比度区域有望让更多真实椭圆被检出。
3. **零 C++ 修改** — 纯 Python wrapper，100% 新增文件。
4. **CNN 迁移路径清晰** — 可演变为 learnable image enhancement preprocessing network。
5. **GeoFilter 的价值经验** — 证明了「删候选」的上限远低于「增候选」。本 idea 从增候选方向切入。

## 6. 实验设计

三组参数（符合 skill 最多 3 组的要求）:

| 参数 | Light | Medium | Strong |
|---|---|---|---|
| CLAHE clipLimit | 2.0 | 3.0 | 4.0 |
| CLAHE tileGridSize | (8,8) | (8,8) | (8,8) |
| Bilateral d | 5 | 7 | 9 |
| Bilateral sigmaColor | 50 | 75 | 100 |
| Bilateral sigmaSpace | 50 | 75 | 100 |

评估方法:
```text
原始图像 → Python 增强 → 临时目录 → aamed_demo.exe (修复版) → .fled.txt → aamed_eval.exe
```

对比基线: 修复版 detector（含 isCombValid 修复，FMeasure=0.530843）在原始图像上的结果。
