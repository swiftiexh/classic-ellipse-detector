# 多尺度图像金字塔椭圆检测 (FPN-style Multi-Scale Ellipse Detection)

## 一句话总结

对输入图像做 2×/3× 上采样，各自跑 AAMED，将检测结果回映射到原图坐标，通过**跨尺度严格共识融合**筛选出小椭圆候选，补充到 baseline 结果中。

**零改动 AAMED 核心逻辑。**

## 效果

| 指标 | Baseline (1×) | +多尺度融合 | 提升 |
|---|---|---|---|
| Precision | 0.7791 | 0.6552 | -0.124 |
| Recall | 0.4026 | 0.4584 | **+0.056** |
| FMeasure | 0.5308 | 0.5394 | **+0.0086** |
| 小椭圆 Recall (<15px) | 0.1278 | 0.2511 | **×2.0** |
| 被挤掉的 baseline GT | — | **0** | 无副作用 |

数据集：Prasad (198 张, 1165 GT, 其中 454 个小椭圆)

## 算法思路

小椭圆的边缘太短/太弱，在原图分辨率下 AAMED 的弧分割无法形成有效弧段。放大图像后边缘变长、变强 → AAMED 能检测到 → 坐标缩回原图 → 跨尺度一致性去噪。

```
原图 1× → AAMED(T=0.77) → baseline detections ──────────────┐
                                                              │
原图 → resize 2× → AAMED(T=0.77) → 坐标÷2 ←──┐               │
原图 → resize 2× → AAMED(T=0.72) → 坐标÷2 ←──┤               │
原图 → resize 3× → AAMED(T=0.77) → 坐标÷3 ←──┼── 小椭圆过滤 ──┤
原图 → resize 3× → AAMED(T=0.72) → 坐标÷3 ←──┘   (r<20px)    │
                                                              │
                    跨尺度严格共识融合 ←────────────────────────┘
                    (≥3 分支支持 + 至少覆盖 2× 和 3×)
                                 │
                         最终检测结果
```

## 文件结构

```
multi_scale_fpn/
├── README.md           ← 本文件
├── ALGORITHM.md        ← 精确算法描述（含伪代码）
├── CPP_INTEGRATION.md  ← C++ 集成指南
├── reference/
│   └── run_scc_small_fpn_eval.py  ← Python 参考实现
└── results/
    ├── summary.json    ← 完整实验数据
    └── report.md       ← 实验报告
```

## 给组长的关键信息

1. **AAMED 核心不需要改**。Canny、FSA、弧分组、验证、聚类都不动。
2. **只需要加一层薄封装**：内部上采样 → 调用现有 `run_FLED` → 坐标回映射 → 跨尺度融合。
3. **Python 参考实现**在 `reference/run_scc_small_fpn_eval.py`，所有逻辑均可逐行翻译为 C++。
4. **融合算法伪代码**在 `ALGORITHM.md`，精确到每一步的输入输出和阈值。
5. **C++ 具体改动清单**在 `CPP_INTEGRATION.md`。
