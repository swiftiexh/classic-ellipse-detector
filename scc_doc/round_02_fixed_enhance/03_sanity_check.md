# SCC-Enhance Sanity Check — deepseek 分支

生成时间：2026-05-30
分支：deepseek

## 1. 测试目标

验证 SCC-Enhance 预处理模块：
1. 能在少量样本上完整运行
2. 增强后图像能被 detector 正确读取
3. 输出 `.fled.txt` 能被 `aamed_eval.exe` 正确评估
4. 不会产生明显全空输出或异常耗时

## 2. 测试样本

Prasad 前 10 张图（与 SCC-GeoFilter sanity 相同的 10 张）：
```
002_0038.jpg, 003_0144.jpg, 008_0132.jpg, 011_0018.jpg, 011_0042.jpg,
011_0100.jpg, 012_0001.jpg, 012_0033.jpg, 012_0190.jpg, 014_0015.jpg
```

## 3. 运行命令

```powershell
D:\anaconda3\python.exe scc_experiments\run_scc_enhance_sanity.py
```

评估：
```powershell
build\bin\aamed_eval.exe --imagenames output\scc_enhance_sanity\results\imagenames.txt --gt-dir datasets\prasad\gt --results-dir output\scc_enhance_sanity\results --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\scc_enhance_sanity_eval.txt
```

## 4. Sanity 结果

输出文件检查：
- 10 个 `.fled.txt` 全部生成
- 增强后图像在 `output/scc_enhance_sanity/images/`

检测统计：
```
images=10, total_detections=37
```

评估输出：
```
Images: 10
PositiveMatches: 20
DetectedCount: 37
GroundTruthCount: 52
Precision: 0.540541
Recall: 0.384615
FMeasure: 0.449438
AverageDetectedTimeMs: 15.550610
```

对比（sanity 10 张图子集）：

| 指标 | 原始 Baseline | SCC-Enhance Light | Δ |
|---|---|---|---|
| PositiveMatches | 16 | 20 | +4 |
| DetectedCount | 31 | 37 | +6 |
| Precision | 0.516129 | 0.540541 | +0.0244 |
| Recall | 0.307692 | 0.384615 | +0.0769 |
| FMeasure | 0.385542 | 0.449438 | +0.0639 |

注：原始 baseline 使用的是未修复的 detector。本文 sanity 使用的是修复版 detector（含 isCombValid 修复）。后续完整实验时将以修复版 detector 在原始图像上的结果作为新 baseline 进行公平对比。

## 5. 是否通过

| 检查项 | 结果 | 说明 |
|---|---|---|
| 程序能完整运行 | 通过 | 10 张图全部处理完成 |
| 输出格式可评估 | 通过 | `aamed_eval.exe` 正常读取 |
| 输出文件数量正确 | 通过 | 10 个 `.fled.txt` |
| 是否全空输出 | 通过 | 37 个候选均检测到 |
| 是否明显耗时异常 | 通过 | 增强+检测均在合理时间内 |
| 是否破坏 baseline | 通过 | 输出到独立目录 `output/scc_enhance_sanity/` |

通过。进入完整实验阶段。
