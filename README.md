# AAMED_OpenCV

`AAMED_OpenCV` 是一个基于 OpenCV 的纯 C++ 椭圆检测工程，改写自公开的 AAMED（Arc Adjacency Matrix-Based Fast Ellipse Detection）仓库。

当前仓库主要提供：

- `aamed_demo`：单图或数据集批量检测
- `aamed_eval`：检测结果评测
- 调试导出：边缘轮廓、DP 轮廓、FSA 弧段、邻接矩阵、最终椭圆结果

## 当前构建方式

环境要求：

- OpenCV 4.x
- CMake 3.20+
- 支持 C++17 的编译器

Windows + MinGW 示例：

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DOpenCV_DIR="D:/OpenCV/opencv-4.5.2/build"
cmake --build build
```

生成的可执行文件：

```txt
build/bin/aamed_demo.exe
build/bin/aamed_eval.exe
```

## 目录结构

```txt
demo/                 单图演示输入
dataset/              数据集、GT 与参考结果
dataset/images/       数据集图像
dataset/gt/           Ground Truth
dataset/AAMED/        参考 AAMED 结果文件
output/single/        当前单图输出
output/dataset/       当前批量输出
src/                  检测器源码
tools/aamed_eval.cpp  评测器源码
```

## 检测程序使用方式

`aamed_demo` 现在不是通过命令行参数配置，而是直接在代码里改配置。

入口文件：

- [`src/main.cpp`](src/main.cpp)

当前默认模式：

- `RunMode::DatasetBatch`

批量模式默认读取：

- `dataset/images/`
- `dataset/imagenames.txt`

批量模式默认输出到：

- `output/dataset/`

运行方式：

```powershell
.\build\bin\aamed_demo.exe
```

## 评测程序使用方式

`aamed_eval` 同样改成了代码内配置，不再通过命令行参数传入。

入口文件：

- [`tools/aamed_eval.cpp`](tools/aamed_eval.cpp)

当前默认评测配置对应的是 `Prasad` 数据集，主要包括：

- `datasetRoot = dataset/`
- `imagesDir = dataset/images/`
- `gtDir = dataset/gt/`
- `resultsDir = output/dataset/`
- `groundTruthSource = PlainTextWithCount`
- `groundTruthConvention = XYRad`
- `gtPrefix = "gt_"`
- `resultFormat = AamedFled`
- `overlapThreshold = 0.8`

运行方式：

```powershell
.\build\bin\aamed_eval.exe
```

## 当前算法默认配置

一些重要的编译期开关在 [`src/definition.h`](src/definition.h) 中。

当前工作区默认值：

- `ADAPT_APPROX_CONTOURS = 1`
- `DEFINITE_ERROR_BOUNDED = 1`
- `FASTER_ELLIPSE_VALIDATION = 0`
- `SELECT_CLUSTER_METHOD = PRASAD_CLUSTER_METHOD`

## 输出内容

单图模式可能生成：

- `detected.png`
- `timing.txt`
- `detections.txt`
- `*.fled.txt`
- `debug/` 中间调试结果

批量模式会生成：

- 每张图对应一个 `*.fled.txt`
- `batch_summary.txt`

## 说明

- 当前仓库为了方便课程项目实验，检测器和评测器都采用“源码内改配置”的方式。
- `dataset/AAMED/` 可作为参考结果目录，用来验证评测器是否正确读取 `.fled.txt`。

## Prasad 结果

论文中给出的 `Prasad` 数据集官方结果为：

| 配置/来源 | Precision | Recall | FMeasure | Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| 论文官方结果 | 77.13 | 39.66 | 52.38 | 4.21 |

在当前实现上的实测结果为：

| 配置/来源 | Precision | Recall | FMeasure | Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| 推荐匹配配置 | 77.78 | 40.26 | 53.05 | 3.74 |
