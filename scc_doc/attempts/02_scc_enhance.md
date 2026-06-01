# Attempt 02: Fixed Baseline and SCC-Enhance

## 目标

该尝试包含两个部分：

1. 修复 detector 中组合有效性状态 `isCombValid` 的作用域问题，并重跑 Prasad 全集，得到 fixed baseline。
2. 在 fixed detector 前增加图像预处理 SCC-Enhance，包括 CLAHE + bilateral filter，再比较三组增强参数。

## 核心代码改动

修改文件：

```text
src/Group.cpp
```

关键修复：

```text
将 isCombValid 从循环外移动到每个 fitComb 候选循环内部初始化。
```

原因：

- 原始代码中 `isCombValid` 在循环外初始化，一旦某个组合被判为 false，后续候选可能继承错误状态。
- 修复后每个候选组合独立判断，避免状态污染。

影响：

- 这是核心 C++ detector 的小范围 bugfix。
- 与 SCC-GeoFilter 不同，它会改变 detector 重新运行时的输出。
- 未覆盖官方 `datasets/prasad/AAMED`，结果输出到独立目录。

## SCC-Enhance 方法

新增脚本：

```text
scc_experiments/scc_enhance.py
scc_experiments/run_fixed_baseline.py
scc_experiments/run_scc_enhance.py
scc_experiments/run_scc_enhance_sanity.py
```

预处理流程：

```text
input image
-> grayscale
-> CLAHE
-> bilateral filter
-> BGR image
-> aamed_demo.exe
```

三组参数：

| 参数组 | CLAHE clip | bilateral d | sigma color | sigma space |
| --- | ---: | ---: | ---: | ---: |
| light | 2.0 | 5 | 50.0 | 50.0 |
| medium | 3.0 | 7 | 75.0 | 75.0 |
| strong | 4.0 | 9 | 100.0 | 100.0 |

## 结果

| 方法 | Precision | Recall | FMeasure | AverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| Official AAMED baseline | 0.771285 | 0.396567 | 0.523810 | 5.146824 |
| Fixed detector baseline | 0.779070 | 0.402575 | 0.530843 | 14.952466 |
| SCC-Enhance light | 0.728296 | 0.388841 | 0.506995 | 15.036036 |
| SCC-Enhance medium | 0.727119 | 0.368240 | 0.488889 | 17.242711 |
| SCC-Enhance strong | 0.783985 | 0.361373 | 0.494712 | 15.353225 |

## 结论

Fixed detector baseline 相比 official AAMED baseline 提升明显：

| 指标 | 差值 |
| --- | ---: |
| Precision | +0.007785 |
| Recall | +0.006008 |
| FMeasure | +0.007033 |
| AverageDetectedTimeMs | +9.805642 |

SCC-Enhance 三组均低于 fixed detector baseline，说明这轮 CLAHE + bilateral 预处理没有带来有效收益。strong 组 Precision 较高，但 Recall 明显下降，FMeasure 也低。

推荐保留：

```text
Group.cpp bugfix
```

不推荐继续：

```text
当前 hard-coded SCC-Enhance 参数搜索
```

## 保留文件

```text
scc_experiments/scc_enhance.py
scc_experiments/run_fixed_baseline.py
scc_experiments/run_scc_enhance.py
scc_experiments/run_scc_enhance_sanity.py
output/scc_fixed_baseline
output/scc_enhance_light_results
output/scc_enhance_medium_results
output/scc_enhance_strong_results
output/scc_*_eval*.txt
scc_doc/logs/round_02_fixed_enhance/scc_fixed_baseline_summary.json
scc_doc/logs/round_02_fixed_enhance/scc_enhance_*_summary.json
```

## 清理说明

增强后的临时图像目录属于中间产物，可删除：

```text
output/scc_enhance_light_images
output/scc_enhance_medium_images
output/scc_enhance_strong_images
```

结果 `.fled.txt` 和评估 `.txt` 保留。
