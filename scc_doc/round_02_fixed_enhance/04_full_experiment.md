# SCC-Enhance 完整实验 — deepseek 分支

生成时间：2026-05-30
分支：deepseek

## 1. 实验设置

- 数据集: Prasad 198 张图片
- Detector: 修复版 aamed_demo（含 isCombValid 修复）
- Evaluator: aamed_eval.exe, IoU >= 0.8
- 对比基线: 修复版 detector + 原始图像（不增强）
- 参数组数: 3 组（Light / Medium / Strong）

## 2. 方法

SCC-Enhance = CLAHE（局部对比度增强）+ Bilateral Filter（保边去噪），三组参数：

| 参数 | Light | Medium | Strong |
|---|---|---|---|
| CLAHE clipLimit | 2.0 | 3.0 | 4.0 |
| CLAHE tileGridSize | (8,8) | (8,8) | (8,8) |
| Bilateral d | 5 | 7 | 9 |
| Bilateral sigmaColor | 50 | 75 | 100 |
| Bilateral sigmaSpace | 50 | 75 | 100 |

## 3. 运行命令

生成结果：
```powershell
D:\anaconda3\python.exe scc_experiments\run_scc_enhance.py
```

评估：
```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_enhance_light_results --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_enhance_light_eval.txt
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_enhance_medium_results --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_enhance_medium_eval.txt
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_enhance_strong_results --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_enhance_strong_eval.txt
```

## 4. 实验结果

### 主表

| 方法 | Precision | Recall | FMeasure | PositiveMatches | DetectedCount |
|---|---|---|---|---|---|
| Fixed Baseline (原始图像) | 0.779070 | 0.402575 | 0.530843 | 469 | 602 |
| SCC-Enhance Light | 0.728296 | 0.388841 | 0.506995 | 453 | 622 |
| SCC-Enhance Medium | 0.727119 | 0.368240 | 0.488889 | 429 | 590 |
| SCC-Enhance Strong | 0.783985 | 0.361373 | 0.494712 | 421 | 537 |

### Delta 表

| 方法 | Delta Precision | Delta Recall | Delta FMeasure | Delta PosMatches |
|---|---|---|---|---|
| SCC-Enhance Light | -0.050774 | -0.013734 | -0.023848 | -16 |
| SCC-Enhance Medium | -0.051951 | -0.034335 | -0.041954 | -40 |
| SCC-Enhance Strong | +0.004915 | -0.041202 | -0.036131 | -48 |

### 完整历史对比

| 方法 | Precision | Recall | FMeasure | 说明 |
|---|---|---|---|---|
| Original Baseline | 0.771285 | 0.396567 | 0.523810 | 原始 detector |
| Fixed Baseline | 0.779070 | 0.402575 | 0.530843 | isCombValid 修复 |
| SCC-Enhance Light | 0.728296 | 0.388841 | 0.506995 | 修复 + CLAHE Light |
| SCC-Enhance Medium | 0.727119 | 0.368240 | 0.488889 | 修复 + CLAHE Medium |
| SCC-Enhance Strong | 0.783985 | 0.361373 | 0.494712 | 修复 + CLAHE Strong |

## 5. 分析

### 核心发现

**SCC-Enhance 在所有三组参数下均未超越 Fixed Baseline（FMeasure 0.530843）。**

- Light: Precision 和 Recall 双双下降。检测数 +20（602→622），但 PositiveMatches -16（469→453），增加的检测全是误检。
- Medium: 效果最差，FMeasure -0.042。
- Strong: Precision 略微上升（+0.005），但 Recall 大幅下降（-0.041）。

### 原因分析

1. **CLAHE 与 Adaptive Canny 功能重叠** — detector 内部的 Adaptive Canny 已做局部自适应阈值，CLAHE 再增强局部对比度导致边缘过检，噪声边缘被 Canny 捕获为虚假弧段。

2. **Bilateral + GaussianBlur 叠加** — detector 在 Canny 前硬编码了 GaussianBlur。Bilateral 保边后的图像再被 Gaussian 平滑，保边效果丧失。

3. **CLAHE 平坦区噪声放大** — 自然场景中均匀区域经 CLAHE 后噪声被放大→虚假边缘→虚假弧段→误检椭圆。

4. **Strong 的 Precision 最高但 Recall 最低** — 更强的 bilateral 滤波去除了噪声但也抹掉了弱椭圆边界。

### 与 GeoFilter 的对比

| 实验 | 策略 | 最佳 FMeasure Delta | 改进类型 |
|---|---|---|---|
| SCC-GeoFilter (scc 分支) | 后处理删候选 | +0.000351 | Post-processing |
| isCombValid fix (scc 分支) | 修复核心 bug | +0.007033 | Core fix |
| SCC-Enhance (deepseek 分支) | 预处理增强 | -0.023848 | Pre-processing |

## 6. 结论

- SCC-Enhance 完整实验结果为负面：三组参数均未超越 Fixed Baseline
- 手工设计的预处理难以精确匹配特定 detector 的内部边缘检测策略
- **但 CNN 迁移价值依然成立** — 可学习的 preprocessing 网络可以自动学习适合特定 detector 的增强策略，而非手工设计 CLAHE 参数
- 预处理的正确方式可能是替换 detector 内部的 GaussianBlur 为 Bilateral Filter，但这需修改 C++ 源码
