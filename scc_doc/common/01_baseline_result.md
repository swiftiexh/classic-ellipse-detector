# baseline 运行结果

生成时间：2026-05-30

## 1. baseline 构建情况

README 推荐使用 CMake 构建：

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DOpenCV_DIR="<OpenCV_DIR>"
cmake --build build
```

本机实际情况：

- `cmake --version` 失败，原因是 `cmake` 不在 PATH 中。
- `g++` 可用，但本机 OpenCV 是 VS/VC16 版本。
- 通过 Visual Studio 2022 Community 的开发环境脚本加载 MSVC 后，手工编译成功。
- OpenCV 动态库运行路径需要加入 PATH：

```text
E:\OPENCV\opencv\build\x64\vc16\bin
```

本轮生成的可执行文件：

```text
build\bin\aamed_demo.exe
build\bin\aamed_eval.exe
```

编译警告：

```text
warning C4819: OpenCV 头文件包含当前代码页 936 无法表示的字符
```

该警告未阻止构建。

## 2. 单图 baseline 检测

数据集：

```text
data/
```

输入图像：

```text
data\images\002_0038.jpg
```

运行命令：

```powershell
cmd /c 'set PATH=E:\OPENCV\opencv\build\x64\vc16\bin;%PATH% && build\bin\aamed_demo.exe --input data\images\002_0038.jpg --output-dir output\baseline_single --export-debug'
```

终端输出：

```text
Pre-Processing: 15.8171 ms.
Arc Segmentation: 0.1069 ms.
Arc Grouping: 0.2245 ms.
Ellipse Fitting: 0.2163 ms, 13.00 times, 0.0166 ms/time
Ellipse Validation: 0.0982 ms, 11.00 times, 0.0089 ms/time
Ellipse Cluster: 0.0006 ms.
Total: 16.4636 ms.
Input: data\images\002_0038.jpg
Detections: 1
Saved image: output\baseline_single\detected.png
Saved results: output\baseline_single\002_0038.jpg.fled.txt
Saved debug artifacts: output\baseline_single\debug
```

输出文件：

| 文件 | 说明 |
| --- | --- |
| `output\baseline_single\detected.png` | 检测可视化图 |
| `output\baseline_single\002_0038.jpg.fled.txt` | AAMED `.fled.txt` 格式检测结果 |
| `output\baseline_single\detections.txt` | 检测表格 |
| `output\baseline_single\timing.txt` | 各阶段耗时 |
| `output\baseline_single\debug\` | 中间调试图与文本 |

检测结果文件内容：

```text
16.4636
1 119.578 118.135 187.007 188.853 -25.1958
```

解释：

- 第一行为总检测耗时，单位 ms。
- 后续每行为一个检测椭圆。
- 当前检测出 1 个椭圆。

## 3. 单图 baseline 评估

评估命令：

```powershell
build\bin\aamed_eval.exe --dataset-root data --results-dir output\baseline_single --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\baseline_single_eval_result.txt
```

评估输出：

```text
Images: 1
PositiveMatches: 1
DetectedCount: 1
GroundTruthCount: 1
Precision: 1.000000
Recall: 1.000000
FMeasure: 1.000000
AverageDetectedTimeMs: 16.463600
```

指标表：

| 指标 | 数值 |
| --- | ---: |
| Images | 1 |
| PositiveMatches | 1 |
| DetectedCount | 1 |
| GroundTruthCount | 1 |
| Precision | 1.000000 |
| Recall | 1.000000 |
| FMeasure | 1.000000 |
| AverageDetectedTimeMs | 16.463600 |

注意：

- 当前评估脚本不输出 AP_0.5、AP_0.75、AP10_0.5、AP10_0.75。
- 不能把 `FMeasure` 等同于 AP。

## 4. Prasad 官方结果评估

本轮没有批量重跑 `aamed_demo` 生成 Prasad 全集结果，因为当前原始入口只支持单图输入，项目中未发现批量 runner。

为了确认评估工具和 Prasad 数据结构可用，本轮只评估了数据集中已有的官方 AAMED 结果：

```text
datasets\prasad\AAMED
```

评估命令：

```powershell
build\bin\aamed_eval.exe --dataset-root datasets\prasad --results-dir datasets\prasad\AAMED --gt-prefix gt_ --gt-format plain_rad --result-format aamed_fled --overlap 0.8 --report output\prasad_official_aamed_eval_result.txt
```

评估输出：

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

指标表：

| 指标 | 数值 |
| --- | ---: |
| Images | 198 |
| PositiveMatches | 462 |
| DetectedCount | 599 |
| GroundTruthCount | 1165 |
| Precision | 0.771285 |
| Recall | 0.396567 |
| FMeasure | 0.523810 |
| AverageDetectedTimeMs | 5.146824 |

重要说明：

- 这不是本轮重跑 detector 的 Prasad baseline。
- 这是对 `datasets\prasad\AAMED` 中已有 `.fled.txt` 文件的评估。
- 后续如果需要真实重跑 Prasad baseline，应先新增批量运行 wrapper，再统一评估输出目录。

## 5. 是否跑通

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 单图 detector baseline | 跑通 | `data\images\002_0038.jpg` 输出 1 个检测 |
| 单图 eval baseline | 跑通 | Precision / Recall / FMeasure 均为 1.0 |
| Prasad 已有结果 eval | 跑通 | 成功评估 198 个已有结果文件 |
| Prasad 全集 detector 重跑 | 未执行 | 当前缺少批量 runner，本阶段不新增实现 |

## 6. 报错与原因

侦察过程中遇到的非阻塞问题：

1. `cmake --version` 失败：

```text
cmake : 无法将“cmake”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。
```

原因：`cmake` 不在当前 PATH 中。

2. 普通 PowerShell 中直接调用 `cl` 编译失败：

```text
fatal error C1083: 无法打开包括文件: “iostream”: No such file or directory
fatal error C1083: 无法打开包括文件: “vector”: No such file or directory
```

原因：普通 shell 未加载 MSVC 标准库 include 路径。使用 Visual Studio 的 `VsDevCmd.bat` 后解决。

3. 第一次 MSVC 宏定义转义错误：

```text
src\main.cpp(14): error C2065: “E”: 未声明的标识符
```

原因：`AAMED_OPENCV_PROJECT_ROOT` 字符串宏的命令行引号未正确转义。修正为带转义引号的宏定义后编译成功。

## 7. 可视化结果

已生成可视化图：

```text
output\baseline_single\detected.png
```

已生成调试中间结果目录：

```text
output\baseline_single\debug
```

本阶段未筛选失败样例；失败样例分析属于后续完整实验阶段。
