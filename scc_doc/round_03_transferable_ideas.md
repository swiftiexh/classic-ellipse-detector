# Round 03: 可迁移到 CNN 的 classic-ellipse-detector 改进 idea

生成时间：2026-05-31

## 0. 当前项目判断

当前项目是 C++ / OpenCV 版 AAMED 椭圆检测流程：

```text
Gaussian + adaptive Canny
-> contours
-> DP approximation
-> FSA arc segmentation
-> arc grouping
-> ellipse fitting
-> fastValidation score
-> ellipse NMS / clustering
```

已完成尝试：

| 方法 | Precision | Recall | FMeasure | 结论 |
| --- | ---: | ---: | ---: | --- |
| Official AAMED baseline | 0.771285 | 0.396567 | 0.523810 | 原始基线 |
| SCC-GeoFilter strict | 0.776094 | 0.395708 | 0.524161 | hard filter 收益极小 |
| Fixed detector baseline | 0.779070 | 0.402575 | 0.530843 | `Group.cpp` bugfix 最有效 |
| SCC-Enhance light/medium/strong | 低于 fixed baseline | 低于 fixed baseline | 低于 fixed baseline | 不建议继续硬编码预处理 |

因此下一步不建议继续堆 CLAHE / bilateral 这类外部预处理，也不建议继续加严 hard geometry filter。更建议做候选级 soft score、覆盖率、top-k 选择、局部几何 refine，这些都很容易迁移到 CNN。

## 1. 最推荐：Top-K + GeoScore 候选重排

### 为什么适合本项目

`src/Group.cpp::BiDirectionVerification` 当前逻辑是：

1. 对 `fitComb` 逐个候选尝试拟合。
2. 第一个通过 `fastValidation` 的候选直接写入 `detEllipses`。
3. 然后 `break`。

代码里原本已有“记录 max_score / max_ellipse”的注释块，但当前没有启用。也就是说，项目现在可能不是选当前 root arc 下最好的椭圆，而是选第一个合法椭圆。

### classic 版本做法

把“第一个 valid 就 break”改成：

```text
遍历候选前 K 个或全部候选
-> 对通过 fastValidation 的候选计算 composite score
-> 选择分数最高者
-> 标记对应 arcs grouped
```

推荐 score：

```text
score = fastValidationScore
      + w1 * angularCoverage
      + w2 * edgeHitRatio
      - w3 * outOfImagePenalty
      - w4 * eccentricityPenalty
```

第一版可以不训练，手调权重即可。

### CNN 迁移方式

迁移成 proposal reranking / candidate quality head。CNN 主干输出椭圆候选后，用几何特征、边界特征、IoU 监督训练一个质量分数。它等价于把 classic detector 的候选选择逻辑变成可学习的 confidence calibration。

### 预期收益

优先级最高。实现简单，且直接瞄准当前代码的明显选择偏差。预计主要提升 Precision 和 IoU=0.8 下的 FMeasure，Recall 通常不会像 hard filter 那样明显掉。

## 2. 强推荐：Angular Coverage / Quadrant Coverage 验证

### 为什么适合本项目

当前 `src/Validation.cpp::fastValidation` 已经沿椭圆轮廓采样，并计算边缘命中和梯度一致性。但它主要是聚合分数，缺少“覆盖分布”的约束：一个局部强边缘可能给出不错分数，却不是完整椭圆；或者真实遮挡椭圆虽然不完整，但覆盖角度分布更合理。

2024 年 Pattern Recognition 的 quadrant representation / top-down fitting 工作明确使用 arc coverage 与 angular coverage 来提高碎裂、遮挡、扁椭圆场景下的精度。

### classic 版本做法

在 `fastValidation` 采样循环里增加角度 bin：

```text
VALIDATION_NUMBER samples
-> 统计每个 angle bin 是否有 edge hit
-> angularCoverage = covered_bins / valid_bins
-> maxGap = 最大连续未命中角度长度
-> quadrantCoverage = 4 个象限覆盖数量
```

然后作为 soft score，不要第一版就 hard reject：

```text
score += w_cov * angularCoverage - w_gap * maxGapRatio
```

### CNN 迁移方式

迁移成 boundary mask / polar ring loss / contour coverage loss。CNN 不只回归 `(cx, cy, a, b, theta)`，还要求预测的椭圆边界在 polar bins 上被边缘或 mask 支持。

### 预期收益

比纯轴比/面积 filter 更有用，尤其适合减少复杂背景 false positive。实现难度低，因为采样循环已经存在。

## 3. 强推荐：通过支持点做一次轻量 robust refit

### 为什么适合本项目

项目评估阈值是 overlap 0.8，参数精度很重要。当前拟合来自弧段累积矩阵，`fastValidation` 已经能找到椭圆采样点附近的真实边缘点，但验证后没有用这些支持点反过来 refine 参数。

### classic 版本做法

对通过 `fastValidation` 的候选：

```text
收集轮廓采样附近命中的 edge points
-> 用 OpenCV fitEllipseAMS / fitEllipseDirect 或一次加权最小二乘重拟合
-> 再跑 fastValidation
-> 若 score / overlap-like proxy 变好则替换
```

第一版可以只对候选做一次 refit，控制耗时。

### CNN 迁移方式

迁移成 geometric refinement head 或 post-refinement module。CNN 先粗回归椭圆，再用边界采样点 / mask feature 做二阶段 refine。与 Mask R-CNN 后接 ellipse fitting、Ellipse R-CNN 的 refined region 思路一致。

### 预期收益

主要提升定位精度，适合 IoU=0.8 评估。实现不复杂，但要小心 outlier，因此建议配合 angular coverage 和支持点数量阈值。

## 4. 推荐：Vertex-aware 局部支持分数

### 为什么适合本项目

椭圆四个顶点附近对轴长、角度很敏感。EDNet / model-aware ellipse detection 这类 CNN 方法把四个 ellipse vertices 作为辅助任务，目的是让网络更关注几何关键点，改善小椭圆和遮挡椭圆。

### classic 版本做法

从候选椭圆参数计算四个顶点：

```text
major axis endpoints
minor axis endpoints
```

在每个顶点附近取小窗口，计算：

```text
vertexEdgeSupport
vertexGradientConsistency
```

加入 `GeoScore`：

```text
score += w_v * mean(top2_or_top3_vertex_support)
```

不要要求 4 个顶点全有边缘，因为遮挡会误伤 Recall。

### CNN 迁移方式

迁移为四顶点 heatmap / auxiliary regression head。训练标签可以直接由 GT ellipse 自动生成，不需要额外标注。

### 预期收益

实现简单，解释性强。对角度和轴长错误的 false positive 有帮助。

## 5. 中推荐：Soft Ellipse-NMS / score-aware clustering

### 为什么适合本项目

当前项目有 `EllipseNonMaximumSuppression(detEllipses, detEllipseScore, 0.7)` 和旧的 `ClusterEllipses`。如果多个候选高度重叠，直接保留最高分可能受当前 score 偏差影响。

### classic 版本做法

保留 hard NMS 作为 baseline，再增加 soft NMS：

```text
IoU 高但不是完全重复的候选不立刻删除
-> score *= exp(-iou^2 / sigma)
-> 最后按 score 输出
```

也可以用 ellipse IoU 替代当前粗略中心/轴长/角度阈值聚类。

### CNN 迁移方式

直接迁移为 rotated/ellipse Soft-NMS，目标检测里非常常见。

### 预期收益

收益可能中等，适合在 GeoScore 稳定后再做。

## 6. 谨慎推荐：Edge-guided 输入，而不是硬编码图像增强

### 为什么要谨慎

你已经试过 CLAHE + bilateral，结果 FMeasure 下降，说明“直接改变输入图像”会破坏 AAMED 的边缘分布。

但 CNN 文献里 edge fusion / edge-guided module 很有效，例如 ElDet 使用 edge map 作为增强输入，EDNet 使用 LoG-like Edge Detection Module 和 Edge Guided Module。

### classic 版本做法

不改原图，只做旁路特征：

```text
原始 Canny edge map 保留
额外生成 LoG/Sobel/multi-threshold edge map
-> 在 validation score 中查询旁路 edge support
```

第一版不要把旁路 edge map 喂回 `findContours`，只把它当额外验证证据。

### CNN 迁移方式

迁移成 edge auxiliary branch，输入 RGB/gray + edge map，或让网络预测 edge map 后再引导 ellipse head。

### 预期收益

对 CNN 迁移价值高，但 classic 版本要避免再次变成“预处理破坏边缘”的路线。

## 7. 建议实验顺序

| 顺序 | idea | 实现难度 | CNN 迁移性 | 推荐理由 |
| ---: | --- | --- | --- | --- |
| 1 | Top-K + GeoScore 候选重排 | 低 | 高 | 当前代码有直接落点，可能立刻改善 |
| 2 | Angular / quadrant coverage | 低 | 高 | 借用现有 validation 采样循环 |
| 3 | 支持点 robust refit | 中 | 高 | 改善 IoU=0.8 下参数精度 |
| 4 | Vertex-aware 支持分数 | 低-中 | 很高 | GT 自动生成顶点监督 |
| 5 | Soft Ellipse-NMS | 中 | 高 | 等 score 稳定后更有效 |
| 6 | Edge-guided 旁路验证 | 中 | 很高 | 不重复 CLAHE 失败路线 |

## 8. 最小可执行下一轮

建议下一轮只做一个组合，命名为：

```text
SCC-GeoScore-Rerank
```

包含：

1. `BiDirectionVerification` 从 first-valid 改为 best-valid/top-k。
2. `fastValidation` 暴露或内部计算 `edgeHitRatio`、`angularCoverage`、`maxGapRatio`。
3. 先用手调线性 score，不引入训练。
4. 跑 Prasad 198 张图，和 fixed baseline 对比 Precision / Recall / FMeasure / time。

如果这个组合有效，再把它整理成 CNN 迁移模块：

```text
candidate geometry quality head
boundary coverage loss
vertex auxiliary head
```

## 9. 参考来源

- ElDet: An Anchor-free General Ellipse Object Detector, ACCV 2022. https://openaccess.thecvf.com/content/ACCV2022/html/Wang_ElDet_An_Anchor-free_General_Ellipse_Object_Detector_ACCV_2022_paper.html
- Ellipse R-CNN: Learning to Infer Elliptical Object from Clustering and Occlusion, arXiv 2020. https://arxiv.org/abs/2001.11584
- Model-aware ellipse detection via parametric correlation learning, Signal Processing 2026. https://www.sciencedirect.com/science/article/pii/S0165168425002567
- A high-precision ellipse detection method based on quadrant representation and top-down fitting, Pattern Recognition 2024. https://www.sciencedirect.com/science/article/pii/S0031320324003546
- Nonlinear circumference-based robust ellipse detection in low-SNR images, Image and Vision Computing 2024. https://www.sciencedirect.com/science/article/abs/pii/S0262885624000726
- Deep Hough Transform for Semantic Line Detection, ECCV/PAMI. https://arxiv.org/abs/2003.04676
