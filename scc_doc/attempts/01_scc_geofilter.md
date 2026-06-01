# Attempt 01: SCC-GeoFilter

## 目标

在不修改 C++ detector 的前提下，对已有 baseline `.fled.txt` 结果做独立几何后处理。

输入：

```text
datasets/prasad/AAMED/*.fled.txt
datasets/prasad/images/
```

输出：

```text
output/scc_geofilter_prasad_loose
output/scc_geofilter_prasad_medium
output/scc_geofilter_prasad_strict
```

## 方法

SCC-GeoFilter 解析 `.fled.txt`：

- 第一行保留为 detector 原始耗时。
- 后续每行保留 AAMED 原始 6 字段格式。
- 不引入 score，因为文件中没有 confidence 字段。

几何规则：

1. 轴长必须为正。
2. 轴比不能超过阈值。
3. 面积比例不能过小或过大。
4. 中心不能明显越界。
5. 旋转外接框不能明显越界。

## 参数组

| 参数组 | max_axis_ratio | min_area_ratio | max_area_ratio | center_margin_ratio | bbox_margin_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| loose | 10.0 | 0.0002 | 1.50 | 0.10 | 0.35 |
| medium | 6.0 | 0.0005 | 1.10 | 0.05 | 0.20 |
| strict | 4.0 | 0.0010 | 0.85 | 0.00 | 0.10 |

## 结果

| 方法 | Precision | Recall | FMeasure | AverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| Official AAMED baseline | 0.771285 | 0.396567 | 0.523810 | 5.146824 |
| SCC-Loose | 0.772575 | 0.396567 | 0.524107 | 5.146824 |
| SCC-Medium | 0.772575 | 0.396567 | 0.524107 | 5.146824 |
| SCC-Strict | 0.776094 | 0.395708 | 0.524161 | 5.146824 |

最佳组：

```text
SCC-Strict
```

相对 official baseline：

| 指标 | 差值 |
| --- | ---: |
| Precision | +0.004809 |
| Recall | -0.000859 |
| FMeasure | +0.000351 |
| AverageDetectedTimeMs | +0.000000 |

## 结论

SCC-GeoFilter 作为 hard filter 可删除少量误检，使 Precision 和 FMeasure 小幅提升，但 strict 组也删除了至少一个正匹配候选，导致 Recall 下降。

下一步不建议继续加强 hard threshold，更建议迁移为：

```text
soft geometry score / re-ranking
```

## 保留文件

```text
scc_experiments/scc_geofilter.py
scc_experiments/run_scc_geofilter_prasad.py
scc_experiments/run_scc_geofilter_sanity.py
scc_experiments/visualize_scc_cases.ps1
output/scc_geofilter_prasad_loose
output/scc_geofilter_prasad_medium
output/scc_geofilter_prasad_strict
output/scc_geofilter_*_eval_result.txt
scc_doc/figures/*.png
scc_doc/logs/round_01_geofilter/scc_geofilter_*.json
```
