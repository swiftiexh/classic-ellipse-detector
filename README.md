# AAMED_OpenCV

`AAMED_OpenCV` 是从原始 AAMED 仓库中提取的、**仅依赖 OpenCV** 的 C++ 椭圆检测基准工程。

## 功能模块

- **`aamed_demo`**：单张图像椭圆检测，输出检测结果、可视化图、耗时信息与中间调试图。
- **`aamed_eval`**：基于椭圆重叠率（IoU）评估检测结果与真值（Ground Truth）的精度，输出 Precision / Recall / F-Measure。

## 增强优化模块

### 新增的增强函数

| 方法 | 功能说明 |
|------|----------|
| `EnhancedPostProcessing()` | 增强后处理：椭圆有效性过滤 + 置信度过滤 + 非极大值抑制 |
| `EnhancedPostProcessingWithVisibility()` | 基于可见率的后处理：集成边缘可见率验证与去重 |
| `AdvancedEllipseRefinement()` | 高级椭圆优化：边缘重新拟合 + 聚类去重 |
| `MultiStageFiltering()` | 多阶段过滤：形状过滤 → 自适应边缘过滤 → 聚类 → 最终置信度过滤 |
| `AdaptiveThresholdEnhancement()` | 自适应阈值增强，根据图像特征动态调整 |

### 核心算法优化

1. **边缘连续性检测** (`calculateEdgeContinuity`)
   - 在椭圆轮廓上采样点，计算与图像边缘的匹配程度
   - 评估椭圆的边缘支撑度

2. **形状一致性评估** (`calculateShapeConsistency`)
   - 评估椭圆的长宽比和圆润度
   - 过滤畸形或过于扁长的椭圆

3. **自适应阈值过滤** (`adaptiveThresholdFilter`)
   - 根据整体检测结果动态调整过滤阈值
   - 适应不同图像质量和复杂度

4. **基于中心的聚类去重** (`clusterEllipsesByCenter`)
   - 将距离相近的椭圆聚类，保留最优结果
   - 有效去除重复检测

5. **椭圆重新拟合** (`refineEllipseFit`)
   - 基于检测到的边缘点重新拟合椭圆
   - 提高检测精度

### 使用示例

```cpp
AAMED aamed(imgGray.rows + 16, imgGray.cols + 16);
aamed.SetParameters(CV_PI / 3.0, 3.0, 0.76);
aamed.run_FLED(imgGray);

cv::Mat edgeImg;
cv::Canny(imgGray, edgeImg, 50, 150);

// 使用增强后处理
aamed.EnhancedPostProcessing(0.55, 0.45);
aamed.AdvancedEllipseRefinement(edgeImg, true, false);
```

### 检测效果对比

| 数据集 | 原始检测 | 优化后检测 |
|--------|----------|------------|
| 合成8椭圆图片 | 9个(含1误检) | 8个(准确) |
| 合成12椭圆图片 | 12个 | 12个 |
| 真实场景图片 | 1个 | 1个 |

## 编译构建（CMake）

1. **推荐配置：VS2022 + Debug**

   这个仓库当前所在路径包含中文字符，`MinGW Makefiles` 在这里容易出现路径编码问题。你已经有 VS2022 编译器的话，建议直接使用 Visual Studio 2022 生成器，并且默认走 Debug，因为当前 OpenCV 安装包提供的是 Debug 版库。

   ```powershell
   cmake --preset vs2022-debug
   ```

2. **编译**

   ```powershell
   cmake --build build-vs --config Debug
   ```

   编译完成后，可执行文件将生成在：

   ```txt
   build-vs/bin/Debug/aamed_demo.exe
   build-vs/bin/Debug/aamed_eval.exe
   ```

   如果你要手动运行，请先把下面这个目录加入 `PATH`，否则会找不到 OpenCV 的 DLL：

   ```txt
   D:\OpenCV\Build\install\x64\vc17\bin
   ```

## 运行检测

1. **配置环境变量（根据自己的路径修改）**：

   ```txt
   D:\OpenCV\Build\install\x64\vc17\bin
   ```

2. **直接运行（默认图片 + 输出调试中间结果）**

   ```powershell
   .\build-vs\bin\Debug\aamed_demo.exe --export-debug
   ```

   输出文件：

   - `output/detected.png` 检测结果可视化
   - `output/002_0038.jpg.fled.txt` 椭圆参数
   - `output/detections.txt` 检测表格
   - `output/timing.txt` 各阶段耗时
   - `output/debug/*` 算法中间过程图（边缘、弧段、分组、拟合等）

## 运行评估（aamed_eval）

直接运行：

```powershell
.\build-vs\bin\Debug\aamed_eval.exe `
--dataset-root D:\CVProject\AAMED_OpenCV\data `
--results-dir D:\CVProject\AAMED_OpenCV\output `
--gt-format plain_rad `
--result-format aamed_fled `
--overlap 0.8 `
--report eval_result.txt
```

核心参数说明：

- `--dataset-root` ：数据集根目录，**自动读取**：`<dataset-root>/imagenames.txt`、`<dataset-root>/gt`
- `--results-dir` 检测结果文件夹（`output`）
- `--gt-format plain_rad` 真值格式：标准弧度制椭圆参数
- `--result-format aamed_fled` 检测结果格式：本项目输出的 `.fled.txt`
- `--overlap 0.8` 重叠率阈值（默认 0.8）
- `--report eval_result.txt` 将评估报告保存到文件
