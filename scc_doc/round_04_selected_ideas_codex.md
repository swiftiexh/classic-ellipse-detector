# Round 04: Codex 筛选的 3 个优先改进方案

生成时间：2026-06-01

本文基于当前 `classic-ellipse-detector` 源码结构和 `scc_doc/round_04_improvement_ideas.md` 中的候选 idea，筛选出最适合当前项目先落地的 3 个方案。目标是：尽量保持 baseline 不变，用参数开关做可控实验，并为后续 CNN / 闭环迁移留下接口。

---

## 0. 当前项目结构判断

当前项目是 C++ / OpenCV 的 AAMED 风格经典椭圆检测器，核心 pipeline 大致为：

```text
输入灰度图
  -> Canny / edge contour
  -> DP 轮廓简化
  -> FSA arc segmentation
  -> arc grouping
  -> DLS ellipse fitting
  -> fastValidation
  -> cluster / NMS-like merge
  -> 输出 .fled.txt
```

关键源码位置：

| 模块 | 文件 / 函数 |
| --- | --- |
| 主流程 | `src/FLED.cpp::FLED::run_FLED()` |
| 命令行入口 | `src/main.cpp` |
| FSA 弧段切分 | `src/Segmentation.cpp::FLED::FSA_Segment()` |
| 弧段排序 | `src/Segmentation.cpp::FLED::SortArcs()` |
| 弧段分组 | `src/FLED.cpp::FLED::Arcs_Grouping()` |
| 双向搜索验证 | `src/Group.cpp::FLED::BiDirectionVerification()` |
| 椭圆验证 | `src/Validation.cpp::FLED::fastValidation()` |
| 输出检测结果 | `src/FLED_drawAndWriteFunctions.cpp::FLED::writeFLED()` |
| 评估脚本 | `tools/aamed_eval.cpp` |

当前最容易安全改进的位置是 `fastValidation()`，因为它位于候选椭圆生成之后，只影响接受 / 拒绝和 score，不会改变前面 FSA / grouping 的候选空间。

---

## 1. 首选方案：A6 尺寸自适应 `T_val`

### 1.1 为什么适合当前项目

当前验证阈值 `_T_val` 是全局固定值，入口默认是 `0.77`：

```cpp
aamed.SetParameters(CV_PI / 3, 3.4, options.T_val);
```

在 `src/Validation.cpp::FLED::fastValidation()` 里，最后使用：

```cpp
if (E_score > inNum * _T_val)
```

这意味着小椭圆、大椭圆、扁椭圆、近圆椭圆都使用同一个阈值。对当前 recall 偏低的问题，A6 是最小改动、最低风险的第一步。

### 1.2 插入文件或函数

建议插入位置：

```text
src/Validation.cpp
  FLED::fastValidation(cv::RotatedRect &res, double *detScore)
```

具体是在函数内已有的 `R`、`r`、`inNum`、`E_score` 都计算完成后，把最终阈值判断替换为“可选的自适应阈值”。

建议新增一个内部 helper：

```text
FLED::adaptiveValidationThreshold(R, r)
```

也可以先不新建函数，第一阶段直接在 `fastValidation()` 末尾局部计算，减少改动面。

### 1.3 是否需要新增参数开关

需要。

建议新增命令行参数：

```text
--adaptive-T-val
--adaptive-T-alpha <float>
--adaptive-T-beta <float>
```

默认值：

```text
adaptive-T-val = false
adaptive-T-alpha = 0.0
adaptive-T-beta = 0.0
```

这样默认行为完全等价于 baseline。

### 1.4 如何保持 baseline 不变

baseline 保持原则：

```text
不开 --adaptive-T-val 时：
  effective_T_val = _T_val

打开 --adaptive-T-val 时：
  effective_T_val = f(_T_val, R, r, alpha, beta)
```

也就是说，默认仍然使用当前的固定阈值：

```cpp
if (E_score > inNum * _T_val)
```

只有显式传参时才走新逻辑。

### 1.5 实验设计

建议先只做 A6，不叠加其他 idea。

实验矩阵：

| 实验名 | 参数 |
| --- | --- |
| baseline | `--T-val 0.77` |
| A6-1 | `--adaptive-T-val --adaptive-T-alpha -0.02 --adaptive-T-beta 0.00` |
| A6-2 | `--adaptive-T-val --adaptive-T-alpha 0.00 --adaptive-T-beta -0.02` |
| A6-3 | `--adaptive-T-val --adaptive-T-alpha -0.02 --adaptive-T-beta -0.02` |
| A6-4 | grid search around best setting |

重点观察：

```text
1. 小椭圆 recall 是否提升；
2. 扁椭圆 recall 是否提升；
3. AP_0.75 / F@0.75 是否明显下降；
4. Time/ms 是否基本不变。
```

### 1.6 如何记录 Time/ms、AP_0.5、AP_0.75

当前 `tools/aamed_eval.cpp` 支持 `--overlap <threshold>`，但严格来说它输出的是 precision / recall / FMeasure，不是真正积分意义上的 AP。

短期建议先记录为：

```text
AP_0.5_proxy  = FMeasure when --overlap 0.5
AP_0.75_proxy = FMeasure when --overlap 0.75
```

表格字段：

| Variant | Time/ms | Precision@0.5 | Recall@0.5 | AP_0.5_proxy | Precision@0.75 | Recall@0.75 | AP_0.75_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | | | | | | | |
| A6 | | | | | | | |

其中 `Time/ms` 使用 `AverageDetectedTimeMs`。

---

## 2. 第二方案：A7 负证据验证

### 2.1 为什么适合当前项目

当前 `fastValidation()` 主要检查候选椭圆边界上是否有边缘支持，也就是正证据。它对“背景纹理刚好在椭圆边界附近形成虚假支持”的 false positive 抑制不足。

A7 的核心是：不仅看候选椭圆边界是否有边缘，还看椭圆内外邻近环带是否存在过多边缘。如果内外环带边缘密度也很高，说明它可能只是复杂纹理，不是真正的椭圆边界。

这个方案适合当前项目，因为它同样只改 validation，不动前面的候选生成。

### 2.2 插入文件或函数

建议插入位置：

```text
src/Validation.cpp
  FLED::fastValidation(cv::RotatedRect &res, double *detScore)
```

当前函数已经计算了：

```text
R, r
angleRot
xyCenter
sample_x / sample_y
grad_x / grad_y
E_score
```

A7 可以复用这些几何量，对缩小和放大的椭圆环带采样：

```text
inner ellipse: R * 0.90, r * 0.90
outer ellipse: R * 1.10, r * 1.10
```

统计：

```text
inner_edge_density
outer_edge_density
boundary_edge_density
negative_score = (inner_edge_density + outer_edge_density) / boundary_edge_density
final_score = positive_score - w_neg * negative_score
```

### 2.3 是否需要新增参数开关

需要。

建议新增：

```text
--negative-validation
--negative-weight <float>
--negative-band-ratio <float>
```

默认值：

```text
negative-validation = false
negative-weight = 0.2
negative-band-ratio = 0.1
```

### 2.4 如何保持 baseline 不变

不开 `--negative-validation` 时：

```text
final_score = E_score
```

打开时：

```text
final_score = E_score - negative_weight * negative_score
```

最终判断仍然保持同样形式：

```text
final_score > inNum * effective_T_val
```

### 2.5 实验设计

A7 建议放在 A6 之后单独实验，避免一开始无法判断收益来源。

实验矩阵：

| 实验名 | 参数 |
| --- | --- |
| baseline | 固定原始参数 |
| A7-1 | `--negative-validation --negative-weight 0.1` |
| A7-2 | `--negative-validation --negative-weight 0.2` |
| A7-3 | `--negative-validation --negative-weight 0.3` |
| A6+A7 | 使用 A6 最优参数 + A7 最优参数 |

重点观察：

```text
1. false positive 是否下降；
2. Precision@0.5 / Precision@0.75 是否提升；
3. Recall 是否被过度牺牲；
4. 椭圆验证耗时是否明显增加。
```

### 2.6 CNN 迁移价值

A7 是后续 Polar Validation CNN 的经典版本雏形。

它可以自然迁移为：

```text
候选椭圆参数
  -> polar / ring band warp
  -> 2D CNN
  -> validation score
```

其中正证据和负证据分别对应 polar patch 中的中心边界带与上下邻近区域。

---

## 3. 第三方案：A1 Multi-theta / Multi-length FSA

### 3.1 为什么适合当前项目

当前 FSA 分割使用固定参数：

```text
theta_fsa = PI / 3
length_fsa = 3.4
```

入口在 `src/main.cpp` 中：

```cpp
aamed.SetParameters(CV_PI / 3, 3.4, options.T_val);
```

FSA 切分发生在：

```text
src/Segmentation.cpp::FLED::FSA_Segment()
```

主流程里每条 DP contour 只切一次：

```cpp
FSA_Segment(dpContours[i]);
```

这对 recall 不友好，因为真实椭圆的边界可能在不同区域有不同曲率和断裂程度。A1 用多组 FSA 参数产生互补弧段，能直接提高候选覆盖率。

### 3.2 插入文件或函数

建议插入位置：

```text
src/FLED.cpp
  FLED::run_FLED()
```

当前流程：

```text
for each dpContour:
  FSA_Segment(dpContour)
```

建议变成可选流程：

```text
if multi_fsa disabled:
  FSA_Segment(dpContour)

if multi_fsa enabled:
  run FSA_Segment with several temporary theta/length settings
  merge arcs into FSA_ArcContours
  deduplicate near-identical arcs
```

涉及的 FSA 函数：

```text
src/Segmentation.cpp::FLED::FSA_Segment()
```

后续仍走现有流程：

```text
SortArcs(FSA_ArcContours)
CreateArcSearchRegion(FSA_ArcContours)
getArcs_KDTrees(FSA_ArcContours)
Arcs_Grouping(FSA_ArcContours)
BiDirectionVerification(...)
```

### 3.3 是否需要新增参数开关

需要。

建议新增：

```text
--multi-fsa
--fsa-profile default
```

第一阶段可以先内置三组参数：

| Profile | theta_fsa | length_fsa | 作用 |
| --- | ---: | ---: | --- |
| relaxed | `PI / 2.5` | `5.0` | 更宽松，倾向产生长弧段 |
| default | `PI / 3` | `3.4` | 当前 baseline |
| strict | `PI / 4` | `2.5` | 更严格，倾向产生短但干净的弧段 |

### 3.4 如何保持 baseline 不变

不开 `--multi-fsa` 时，保持当前单组 FSA：

```text
PI / 3, 3.4
```

打开 `--multi-fsa` 时，才运行额外 profile。

为了避免候选暴涨，需要加去重策略：

```text
如果两个 arc 的起点距离近、终点距离近、长度接近，则只保留更长或质量更高的 arc。
```

### 3.5 实验设计

A1 会改变候选空间，因此建议放在 A6 / A7 之后。

实验矩阵：

| 实验名 | FSA 参数 |
| --- | --- |
| baseline | default |
| A1-relaxed | default + relaxed |
| A1-strict | default + strict |
| A1-full | default + relaxed + strict |
| A1-full + A7 | full multi-FSA + negative validation |

重点观察：

```text
1. Recall@0.5 是否明显提升；
2. Precision 是否下降；
3. FSA arc count 是否暴涨；
4. ArcGroupingMs 是否暴涨；
5. EllipseValidationCount 是否暴涨。
```

### 3.6 CNN 迁移价值

A1 是 Arc Proposal Network 的经典前身。

后续 CNN 可以学习：

```text
contour point sequence
  -> P(start | point)
  -> P(end | point)
  -> arc proposal list
```

这相当于把多阈值 FSA 的人工枚举，替换成可学习的弧段提案器。

---

## 4. 统一实验记录方式

### 4.1 当前 evaluator 的实际含义

`tools/aamed_eval.cpp` 当前通过 `--overlap <threshold>` 计算：

```text
Precision
Recall
FMeasure
AverageDetectedTimeMs
```

它没有根据 detection score 排序，也没有计算 PR 曲线面积，因此严格来说不是 AP。

建议短期记录：

```text
AP_0.5_proxy  = FMeasure at overlap 0.5
AP_0.75_proxy = FMeasure at overlap 0.75
```

中期如果要严格 AP，需要让输出文件保留 `detEllipseScore`，然后 evaluator 按 score 排序积分 PR 曲线。

### 4.2 推荐表格

| Variant | Time/ms | P@0.5 | R@0.5 | AP_0.5_proxy | P@0.75 | R@0.75 | AP_0.75_proxy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | | | | | | | | `T_val=0.77` |
| A6 | | | | | | | | adaptive threshold |
| A7 | | | | | | | | negative evidence |
| A1 | | | | | | | | multi-FSA |
| A6+A7 | | | | | | | | validation-only combo |
| A1+A6+A7 | | | | | | | | full phase-1 combo |

### 4.3 A1 额外记录

A1 会影响候选数量和速度，因此额外记录：

| Variant | FSAArcs | ArcSegmentationMs | ArcGroupingMs | EllipseValidationMs | EllipseValidationCount |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | | | | | |
| A1-relaxed | | | | | |
| A1-strict | | | | | |
| A1-full | | | | | |

当前 `showDetailBreakdown()` 和 `exportDebugArtifacts()` 已经提供了这些字段的基础。

---

## 5. 第一阶段实现顺序建议

推荐顺序：

```text
Phase 1.1: A6 尺寸自适应 T_val
Phase 1.2: A7 负证据验证
Phase 1.3: A1 Multi-FSA
Phase 1.4: A6 + A7 + A1 组合实验
```

最适合作为第一阶段第一项的是：

```text
A6 尺寸自适应 T_val
```

理由：

```text
1. 插入点最小，只影响 Validation.cpp::fastValidation()；
2. 默认关闭即可保持 baseline 完全不变；
3. 不改变候选弧段、分组和拟合，方便归因；
4. Time/ms 基本不会增加；
5. 能快速建立实验表格和参数扫描流程；
6. 后续可以自然叠加 A7，再叠加 A1。
```

---

## 6. 对原始提示词的评价

原始提示词质量较高，因为它明确了：

```text
1. 阅读当前项目；
2. 不修改代码；
3. 从已有候选 idea 中筛选；
4. 结合源码结构说明；
5. 要求输出插入点、参数开关、baseline、实验、指标记录和第一阶段选择。
```

可以进一步优化为：

```text
请阅读当前 classic-ellipse-detector 项目和 scc_doc/round_04_improvement_ideas.md，但不要修改代码。
请只从文档中的已有候选 idea 里筛选 3 个最适合当前代码结构、最适合第一阶段实验的方案。
要求按“可落地性、风险、对 precision/recall/speed 的预期影响、CNN 迁移价值”排序，并结合源码说明：
1. 插入哪个文件或函数；
2. 是否需要新增参数开关，默认值是什么；
3. 如何保证不开开关时 baseline bitwise/behavior 尽量不变；
4. 如何设计 ablation 实验；
5. 如何记录 Time/ms、AP_0.5、AP_0.75，如果当前 evaluator 不是真 AP，请说明 proxy；
6. 哪个 idea 最适合作为第一阶段第一项实现，以及原因。
```

这个版本会更稳定地引导模型输出“工程落地建议”，而不是泛泛讨论 CNN 架构。
