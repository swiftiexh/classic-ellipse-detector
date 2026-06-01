# SCC-GeoFilter 完整实验记录

生成时间：2026-05-30

## 1. 实验目的

本实验在不修改原始 AAMED C++ detector 核心流程的前提下，基于 Prasad 数据集中已有 baseline 输出：

```text
datasets\prasad\AAMED
```

实现一个独立后处理模块 `SCC-GeoFilter`。该模块读取 baseline `.fled.txt` 检测结果，按轻量几何规则删除明显不合法或低质量的椭圆候选，再用项目已有 `aamed_eval.exe` 评估 Precision、Recall、FMeasure 和 AverageDetectedTimeMs。

本实验不实现 AP 指标，不声称 AP 提升。

## 2. baseline 结果

baseline 评估命令：

```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir datasets\prasad\AAMED --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\prasad_official_aamed_eval_result.txt
```

baseline 评估结果：

```text
Images: 198
PositiveMatches: 462
DetectedCount: 599
GroundTruthCount: 1165
Precision: 0.771285
Recall: 0.396567
FMeasure: 0.523810
AverageDetectedTimeMs: 5.146824
```

## 3. `.fled.txt` 格式分析

从 `datasets\prasad\AAMED` 抽样检查到的真实格式示例：

```text
15.1655
1 119.581 118.135 187.005 188.86 -25.0699
```

格式说明：

| 行 / 字段 | 含义 |
| --- | --- |
| 第 1 行 | detector 耗时，单位 ms |
| 后续每行 | 一个检测候选，包含 6 个数值字段 |
| field[0] | 类型标记；`aamed_eval` 中跳过 `2`，普通检测为 `1` |
| field[1] | row-like center 坐标 |
| field[2] | col-like center 坐标 |
| field[3] | 椭圆轴长 / 直径字段 |
| field[4] | 椭圆轴长 / 直径字段 |
| field[5] | 角度，degree |

`aamed_eval.exe` 对 `aamed_fled` 的读取方式为：

```text
timeMs = first line
cx = field[2] + 1
cy = field[1] + 1
a = field[3] / 2
b = field[4] / 2
theta = -field[5] / 180 * pi
```

文件中没有 confidence score，因此 SCC-GeoFilter 只使用几何规则。输出文件保留原始首行耗时和未删除候选的原始 6 字段行，确保仍可被 `aamed_eval.exe` 正确读取。

## 4. SCC-GeoFilter 方法说明

新增脚本：

```text
scc_experiments\scc_geofilter.py
scc_experiments\run_scc_geofilter_prasad.py
```

输入：

```text
datasets\prasad\imagenames.txt
datasets\prasad\images
datasets\prasad\AAMED
```

输出：

```text
output\scc_geofilter_prasad_loose
output\scc_geofilter_prasad_medium
output\scc_geofilter_prasad_strict
```

筛选规则：

1. 删除轴长非正数的候选。
2. 删除长短轴比例超过阈值的候选。
3. 删除面积比例过小或过大的候选。
4. 删除中心点明显超出图像边界的候选。
5. 删除旋转椭圆外接盒明显超出图像边界的候选。

图像尺寸读取使用 Python 标准库解析 JPEG / PNG header，不引入 OpenCV、PIL 或 CNN 依赖。

## 5. 参数设置

| 参数组 | max_axis_ratio | min_area_ratio | max_area_ratio | center_margin_ratio | bbox_margin_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| loose | 10.0 | 0.0002 | 1.50 | 0.10 | 0.35 |
| medium | 6.0 | 0.0005 | 1.10 | 0.05 | 0.20 |
| strict | 4.0 | 0.0010 | 0.85 | 0.00 | 0.10 |

## 6. 运行命令

生成 SCC 结果：

```powershell
D:\anaconda3\python.exe scc_experiments\run_scc_geofilter_prasad.py
```

说明：直接运行 `python scc_experiments\run_scc_geofilter_prasad.py` 在当前 PowerShell 会话失败，报错为：

```text
程序“python.exe”无法运行: 指定的登录会话不存在。可能已被终止。
```

因此本轮使用：

```text
D:\anaconda3\python.exe
```

评估 SCC-Loose：

```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_geofilter_prasad_loose --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_geofilter_prasad_loose_eval_result.txt
```

评估 SCC-Medium：

```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_geofilter_prasad_medium --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_geofilter_prasad_medium_eval_result.txt
```

评估 SCC-Strict：

```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir output\scc_geofilter_prasad_strict --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_geofilter_prasad_strict_eval_result.txt
```

语法检查：

```powershell
D:\anaconda3\python.exe -m py_compile scc_experiments\scc_geofilter.py scc_experiments\run_scc_geofilter_prasad.py
```

## 7. 实验结果

| 方法 | Precision | Recall | FMeasure | AverageDetectedTimeMs | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.771285 | 0.396567 | 0.523810 | 5.146824 | 已有 `datasets\prasad\AAMED` |
| SCC-Loose | 0.772575 | 0.396567 | 0.524107 | 5.146824 | 删除 1 个候选 |
| SCC-Medium | 0.772575 | 0.396567 | 0.524107 | 5.146824 | 删除 1 个候选 |
| SCC-Strict | 0.776094 | 0.395708 | 0.524161 | 5.146824 | 删除 5 个候选 |

相对 baseline 的差值：

| 方法 | ΔPrecision | ΔRecall | ΔFMeasure | ΔAverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| SCC-Loose | +0.001290 | +0.000000 | +0.000297 | +0.000000 |
| SCC-Medium | +0.001290 | +0.000000 | +0.000297 | +0.000000 |
| SCC-Strict | +0.004809 | -0.000859 | +0.000351 | +0.000000 |

按最高 FMeasure 选择主结果：

```text
SCC-Strict
```

## 8. 过滤前后候选数量统计

| 参数组 | 图像数 | 过滤前候选数 | 过滤后候选数 | 删除数 | 保留比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| loose | 198 | 584 | 583 | 1 | 0.998288 |
| medium | 198 | 584 | 583 | 1 | 0.998288 |
| strict | 198 | 584 | 579 | 5 | 0.991438 |

删除原因统计：

| 参数组 | nonpositive_axis | axis_ratio | area_small | area_large | center | bbox | malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loose | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| medium | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| strict | 0 | 1 | 0 | 0 | 0 | 4 | 0 |

SCC 后处理总耗时：

```text
3 组参数共 1157.212 ms
```

注意：`AverageDetectedTimeMs` 来自 `.fled.txt` 首行的 detector 原始耗时。SCC-GeoFilter 没有改写该耗时，因此评估表中的 AverageDetectedTimeMs 与 baseline 相同。后处理耗时单独记录，不混入 detector time。

## 9. 结果文件路径

| 内容 | 路径 |
| --- | --- |
| baseline 评估报告 | `output\prasad_official_aamed_eval_result.txt` |
| SCC-Loose 输出 | `output\scc_geofilter_prasad_loose` |
| SCC-Medium 输出 | `output\scc_geofilter_prasad_medium` |
| SCC-Strict 输出 | `output\scc_geofilter_prasad_strict` |
| SCC-Loose 评估报告 | `output\scc_geofilter_prasad_loose_eval_result.txt` |
| SCC-Medium 评估报告 | `output\scc_geofilter_prasad_medium_eval_result.txt` |
| SCC-Strict 评估报告 | `output\scc_geofilter_prasad_strict_eval_result.txt` |
| 汇总日志 | `scc_doc\logs\round_01_geofilter\scc_geofilter_prasad_summary.json` |
| 命令日志 | `scc_doc\logs\round_01_geofilter\scc_geofilter_commands.txt` |
| 评估输出日志 | `scc_doc\logs\round_01_geofilter\scc_geofilter_eval_results.txt` |

## 10. 初步结论

SCC-GeoFilter 在 Prasad 已有 AAMED 输出上带来了非常小的 FMeasure 提升。最佳组 `strict` 的 FMeasure 从 0.523810 提升到 0.524161，ΔFMeasure 为 +0.000351；Precision 提升 +0.004809，但 Recall 下降 -0.000859。

这说明 hard geometry filter 可以删除少量明显较差候选，但收益有限，并且严格过滤会删除至少一个正匹配。下一轮更值得尝试的是把几何规则改为 soft score 或 re-ranking，而不是继续加强 hard filter。
