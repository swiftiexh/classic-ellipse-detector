# Baseline 四种优化方法实施梳理

本文档用于记录当前 `AAMED_OpenCV` Baseline 上计划尝试的四种优化方法，供后续切换分支时快速查阅。

当前 Baseline 的主流程可概括为：

1. `Canny + findContours`
2. `adaptApproxPolyDP`
3. `FSA_Segment`
4. `Arcs_Grouping`
5. `BiDirectionVerification`
6. `fastValidation`
7. `Cluster / NMS`

结合当前对 `011_0042.jpg`、`031_0038.jpg`、`033_0053.jpg` 的 badcase 分析，主要问题是：

- 遮挡后的大目标只剩残弧，残弧不容易组成正确椭圆
- 部分候选即使组成了，也可能在验证阶段被刷掉
- 内部小圆、局部闭合结构、重复候选有时比真正的大目标更容易留下


## 1. 加权弧图 / 软兼容 AAM

### 方法概述

当前 Baseline 的弧邻接关系本质上是二值的：

- 能连
- 不能连

这对完整轮廓比较有效，但对遮挡后的短弧、残弧比较苛刻。  
本方法的核心思路是把“是否邻接”改成“邻接兼容度”，也就是：

- 不再只给 `-1 / 0 / 1`
- 而是给一个连续分数，表示两段弧有多像属于同一个椭圆

这样做的目的，是让那些“虽然不够完整，但看起来仍可能属于同一椭圆”的弧段先进入候选池，而不是在分组前期被硬过滤掉。

### 主要想解决的问题

- `011_0042.jpg` 中大量球边缘短弧在前期就活不下来
- `031_0038.jpg`、`033_0053.jpg` 中被遮挡切断后的大轮廓残弧较难形成有效组合

### 拟修改文件

- [src/FLED.cpp](/d:/CVProject/AAMED_OpenCV/src/FLED.cpp)
  - `Arcs_Grouping`
  - `getlinkArcs`
  - 末端 root arc 搜索前后的候选组织逻辑
- [src/FLED.h](/d:/CVProject/AAMED_OpenCV/src/FLED.h)
  - `LinkMatrix`
  - `linkArc`
  - `CASE_1`
  - `Group4FAnB1_FBmA1`
  - `Group4CAnB1_FBmA1`
  - `Group4FAnB1_CBmA1`
  - `Group4CAnB1_CBmA1`
- [src/Group.cpp](/d:/CVProject/AAMED_OpenCV/src/Group.cpp)
  - `PosteriorArcsSearch2`
  - `AnteriorArcsSearch`
  - `BiDirectionVerification`
- [src/LinkMatrix.h](/d:/CVProject/AAMED_OpenCV/src/LinkMatrix.h)
  - 若需要把邻接矩阵从 `char` 扩展为连续权值，可能要改这里

### 备注

这条线主要改“候选怎么进入系统”，一般优先影响 `Recall`。  
它是后续三种方法的前置基础之一。


## 2. 可见扇区验证 / 负证据感知验证

### 方法概述

当前 Baseline 的验证更偏向“整圈都应有较好支持”。  
但在遮挡场景下，真实目标天然只会在局部可见，因此这种全周式验证会让遮挡目标吃亏。

本方法的核心思路是：

- 把候选椭圆沿角度划分成多个扇区
- 只在“应该可见”的扇区上重点累计支持
- 对明显被遮挡、或本来就不该出现边缘的扇区降低惩罚
- 若某些本应可见的位置却没有支持，再作为负证据处理

### 主要想解决的问题

- `031_0038.jpg` 中部分遮挡后的候选可能已经形成，但在验证时被刷掉
- `033_0053.jpg` 中大圆外轮廓残缺，导致验证更偏向内部更闭合的小圆
- 让遮挡目标不再因为“整圈不完整”而天然吃亏

### 拟修改文件

- [src/Validation.cpp](/d:/CVProject/AAMED_OpenCV/src/Validation.cpp)
  - `fastValidation`
  - 如有必要，也同步调整 `Validation`
- [src/FLED.h](/d:/CVProject/AAMED_OpenCV/src/FLED.h)
  - 验证相关成员变量
  - 验证辅助缓存
  - 如需增加扇区统计缓存，也会放这里
- [src/definition.h](/d:/CVProject/AAMED_OpenCV/src/definition.h)
  - 若想给新验证逻辑加宏开关，可在这里增加

### 备注

这条线主要改“候选如何通过最后验证”，同样主要影响 `Recall`，并且最贴合“部分遮挡”课题主题。


## 3. 环带支持差分 / 嵌套竞争

### 方法概述

当前 Baseline 在一些图上会出现：

- 真正的大外轮廓因为遮挡只剩残弧
- 反而内部更小、更完整的圆或椭圆先被保留

本方法的核心思路是：

- 当一个大椭圆与内部小椭圆存在嵌套关系时，不仅比较“谁更像椭圆”
- 还要比较“谁更像这个区域真正的主目标”
- 引入外环支持、独占解释度、嵌套冲突等概念，避免小内圈抢走大外轮廓的位置

### 主要想解决的问题

- `033_0053.jpg` 中后排大圆盘外轮廓没立起来，反而内部小圆更容易留下
- `031_0038.jpg` 中复杂轮胎结构里，局部结构与外轮廓存在竞争

### 拟修改文件

- [src/FLED.cpp](/d:/CVProject/AAMED_OpenCV/src/FLED.cpp)
  - 最终候选生成完成后的后处理入口
- [src/Validation.cpp](/d:/CVProject/AAMED_OpenCV/src/Validation.cpp)
  - `ClusterEllipses` 若沿用原聚类框架，可能会在这里扩展
- [src/EllipseNonMaximumSuppression.cpp](/d:/CVProject/AAMED_OpenCV/src/EllipseNonMaximumSuppression.cpp)
  - 若把嵌套竞争合并进 NMS/后处理，也可能改这里
- 可能新增文件
  - `src/NestedCompetition.h`
  - `src/NestedCompetition.cpp`

### 备注

这条线主要改“多个候选都成立时，谁最终留下”。  
它既能减少误检，也能减少“真目标被次级结构替代”的漏检。


## 4. 椭圆 eIoU Soft-NMS / 支持集感知聚类

### 方法概述

当前 Baseline 的后处理主要有两种：

- `PRASAD_CLUSTER_METHOD`
- `OUR_CLUSTER_METHOD`

其中 `OUR_CLUSTER_METHOD` 本质上已经是基于椭圆 overlap 的硬 NMS，但仍然比较粗：

- 重叠过大就直接删
- 不考虑连续降分
- 不考虑多个相近候选之间的软竞争

本方法的核心思路是：

- 用椭圆级 overlap/eIoU 替代粗糙重复保留
- 从“硬删除”改成“Soft-NMS 连续降分”
- 如有必要，再把“支持集感知”加进去，即不仅看参数像不像，还看是否在解释同一批弧段

### 主要想解决的问题

- `031_0038.jpg` 中同一轮胎被重复检出多个相近椭圆
- `033_0053.jpg` 中一些相近局部候选可能需要更平滑地压分或合并

### 拟修改文件

- [src/EllipseNonMaximumSuppression.cpp](/d:/CVProject/AAMED_OpenCV/src/EllipseNonMaximumSuppression.cpp)
  - 当前 `EllipseNonMaximumSuppression` 的主要实现位置
- [src/EllipseNonMaximumSuppression.h](/d:/CVProject/AAMED_OpenCV/src/EllipseNonMaximumSuppression.h)
  - 若新增 Soft-NMS 接口或参数，需要同步改这里
- [src/FLED.cpp](/d:/CVProject/AAMED_OpenCV/src/FLED.cpp)
  - 最终后处理调用入口
- [src/Validation.cpp](/d:/CVProject/AAMED_OpenCV/src/Validation.cpp)
  - 如果把支持集感知聚类并入现有聚类逻辑，可能改这里
- [src/Group.cpp](/d:/CVProject/AAMED_OpenCV/src/Group.cpp)
  - 若后续要保留“候选由哪些弧段组成”的信息，支持集感知部分可能需要从这里取数据

### 备注

这条线主要改“如何去重”，通常优先影响 `Precision`，也有助于让最终结果更干净。


## 建议的分支命名

建议四个方法分别从同一个 Baseline 提交切出独立分支，例如：

- `method-weighted-arc`
- `method-visible-sector-validation`
- `method-nested-competition`
- `method-eiou-softnms`

建议先提交一个明确的 Baseline 快照，再从该提交切出四个分支，避免后续实验串线。


## 当前 Baseline 关键入口文件

后续做任何方法时，优先参考这些文件：

- [src/FLED.cpp](/d:/CVProject/AAMED_OpenCV/src/FLED.cpp)
- [src/FLED.h](/d:/CVProject/AAMED_OpenCV/src/FLED.h)
- [src/Group.cpp](/d:/CVProject/AAMED_OpenCV/src/Group.cpp)
- [src/Validation.cpp](/d:/CVProject/AAMED_OpenCV/src/Validation.cpp)
- [src/EllipseNonMaximumSuppression.cpp](/d:/CVProject/AAMED_OpenCV/src/EllipseNonMaximumSuppression.cpp)
- [src/Segmentation.cpp](/d:/CVProject/AAMED_OpenCV/src/Segmentation.cpp)
- [src/main.cpp](/d:/CVProject/AAMED_OpenCV/src/main.cpp)
- [tools/aamed_eval.cpp](/d:/CVProject/AAMED_OpenCV/tools/aamed_eval.cpp)

