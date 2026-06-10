# AAMED_OpenCV

`AAMED_OpenCV` 是一个基于 OpenCV 的纯 C++ 椭圆检测工程，改写自公开的 AAMED（Arc Adjacency Matrix-Based Fast Ellipse Detection）实现。

当前仓库已经不是单一 baseline，而是一个**统一实验框架**，可以在同一套代码里切换：

- 数据集
- 单图 / 批量模式
- 不同优化方法
- 方法组合

详细实验配置请直接参考 [实验配置说明.md](/d:/CVProject/AAMED_OpenCV/实验配置说明.md:1)。

## 构建

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

## 当前框架能力

当前统一框架支持：

- `baseline`
- `weighted_arc`
- `multi_scale_fpn`
- `small_ellipse_guard`
- 多种方法组合实验

当前已接入的数据集包括：

- `Prasad Images - Dataset Prasad`
- `Concentric Ellipses - Dataset Synthetic`
- `Concurrent Ellipses - Dataset Synthetic`
- `Random Images - Dataset #1`
- `Smartphone Images - Dataset #2`

## 代码入口

主要入口文件：

- [src/main.cpp](/d:/CVProject/AAMED_OpenCV/src/main.cpp:1)
- [tools/aamed_eval.cpp](/d:/CVProject/AAMED_OpenCV/tools/aamed_eval.cpp:1)
- [src/ExperimentConfig.h](/d:/CVProject/AAMED_OpenCV/src/ExperimentConfig.h:1)

其中：

- `aamed_demo` 负责单图或数据集检测
- `aamed_eval` 负责评测
- `ExperimentConfig.h` 负责统一配置实验模式、数据集、方法和参数

## 运行方式

检测：

```powershell
.\build\bin\aamed_demo.exe
```

评测：

```powershell
.\build\bin\aamed_eval.exe
```

也可以通过环境变量覆盖 `ExperimentConfig.h` 中的默认数据集和方法，无需重新编译：

```powershell
$env:AAMED_DATASET = "random"
$env:AAMED_METHODS = "weighted_arc,small_ellipse_guard"
.\build\cmake_build\bin\Release\aamed_demo.exe
.\build\cmake_build\bin\Release\aamed_eval.exe
```

支持的数据集短名包括 `prasad`、`random`、`concentric`、`concurrent` 和 `smartphone`。方法名包括 `baseline`、`weighted_arc`、`multi_scale_fpn` 和 `small_ellipse_guard`。

运行前只需要修改一个文件：

- [src/ExperimentConfig.h](/d:/CVProject/AAMED_OpenCV/src/ExperimentConfig.h:1)

具体如何切换：

- 数据集
- 单图 / 批量模式
- 方法开关
- 方法参数

请参考 [实验配置说明.md](/d:/CVProject/AAMED_OpenCV/实验配置说明.md:1)。

## 输出目录

当前输出目录按数据集和实验标签自动生成，例如：

- `output/prasad/baseline`
- `output/random/weighted_arc`
- `output/prasad/weighted_arc__multi_scale_fpn`
- `output/random/weighted_arc__multi_scale_fpn__small_ellipse_guard`

单图模式输出在：

- `output/single/<experiment_label>/`

批量模式输出在：

- `output/<experiment_label>/`

## 当前状态说明

- `weighted_arc`：当前已知正优化，参数已接入为最佳已知配置
- `multi_scale_fpn`：当前已知正优化，参数已接入为最佳已知配置
- `small_ellipse_guard`：已经接入统一框架，但目前仍是按提供的默认参数接入，尚未完成系统调参与组合验证

## 说明

- 当前仓库采用“源码内统一配置”的方式，不再依赖复杂命令行参数。
- 对于不同数据集的 GT 命名、评测口径和 overlap 阈值，已经统一封装进实验配置。
- `tools/run_benchmark_matrix.ps1` 可自动运行 Prasad 与 Random 的缺失实验矩阵并生成 CSV、Markdown 汇总。
- 更详细的配置说明、参数解释和推荐实验顺序，请参考 [实验配置说明.md](/d:/CVProject/AAMED_OpenCV/实验配置说明.md:1)。
