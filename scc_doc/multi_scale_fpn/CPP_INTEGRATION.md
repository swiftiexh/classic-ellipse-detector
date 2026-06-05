# C++ 集成指南

## 改动原则

- **不改 AAMED 核心**: `Segmentation.cpp` `Group.cpp` `Validation.cpp` `Contours.cpp` 等完全不动
- **仅新增封装层**: 在 `FLED` 类上添加一个方法 + 数据结构
- **向后兼容**: 不传多尺度参数时行为与原来完全一致

## 需要修改的文件

### 1. `src/FLED.h` — 新增数据结构和方法声明

在 `SCCDeferredConfig` 结构体附近（约第 55 行）新增：

```cpp
// ========== 多尺度配置 (新增) ==========
struct MultiScaleConfig
{
    std::vector<int> scales = {2, 3};        // 上采样尺度 (不含1×)
    std::vector<double> thresholds = {0.77, 0.72}; // 每个尺度使用的 T_val
    int fusionMode = 3;                       // 融合模式: 3=严格共识(推荐)
    double fusionIoU = 0.8;                   // 融合 IoU 阈值
    int minBranches = 3;                      // 严格共识最少支持分支数
    double remapRadiusMax = 20.0;             // 回映射后候选半径上限(半轴几何均值)
};

struct ScaleCandidate
{
    int scale;                                // 来源尺度 (2 或 3)
    double tVal;                              // 来源阈值 (0.77 或 0.72)
    cv::RotatedRect ellipse;                  // 已回映射到原图坐标
    double score;                             // AAMED 验证分数
};
```

在 `FLED` 类的 public 区域（约第 165 行 `run_FLED` 声明之后）新增：

```cpp
    // ========== 多尺度检测 (新增) ==========
    void run_FLED_MultiScale(const cv::Mat& imgGray, const MultiScaleConfig& cfg);
    void setMultiScaleConfig(const MultiScaleConfig& cfg) { _ms_cfg = cfg; }

private:
    MultiScaleConfig _ms_cfg;
    std::vector<ScaleCandidate> _ms_candidates;

    // 内部方法
    void _remapCandidates(int scale, std::vector<cv::RotatedRect>& detections,
                          std::vector<ScaleCandidate>& out);
    double _ellipseIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) const;
    void _buildIoUGraph(const std::vector<ScaleCandidate>& candidates,
                        std::vector<std::vector<int>>& graph) const;
    void _findClusters(const std::vector<std::vector<int>>& graph,
                       int N, std::vector<std::vector<int>>& clusters) const;
    int _selectRepresentative(const std::vector<int>& cluster,
                              const std::vector<ScaleCandidate>& candidates,
                              const std::vector<std::vector<int>>& graph) const;
    void _fuseCandidates(std::vector<cv::RotatedRect>& fused);
```

### 2. `src/FLED.cpp` — 实现多尺度方法 (~200 行)

```cpp
void FLED::run_FLED_MultiScale(const cv::Mat& imgGray, const MultiScaleConfig& cfg)
{
    _ms_cfg = cfg;
    _ms_candidates.clear();

    // 1. Baseline: 1× 检测 (存入 detEllipses)
    run_FLED(imgGray);
    std::vector<cv::RotatedRect> baseline = detEllipses;

    // 2. 各尺度分支检测
    std::vector<std::vector<ScaleCandidate>> branchCandidates;
    for (int s : cfg.scales)
    {
        for (double t : cfg.thresholds)
        {
            // 2.1 上采样
            cv::Mat upscaled;
            cv::resize(imgGray, upscaled,
                       cv::Size(imgGray.cols * s, imgGray.rows * s),
                       0, 0, cv::INTER_CUBIC);

            // 2.2 运行 AAMED
            SetParameters(CV_PI / 3, 3.4, t);
            run_FLED(upscaled);

            // 2.3 回映射 + 过滤
            std::vector<ScaleCandidate> remapped;
            _remapCandidates(s, detEllipses, remapped);

            // 2.4 小椭圆过滤
            for (auto& c : remapped)
            {
                double a = c.ellipse.size.height * 0.5;  // 半轴
                double b = c.ellipse.size.width  * 0.5;
                double r = std::sqrt(a * b);
                if (r < cfg.remapRadiusMax)
                    _ms_candidates.push_back(c);
            }

            branchCandidates.push_back(remapped);
        }
    }

    // 3. 跨尺度融合
    std::vector<cv::RotatedRect> fused;
    _fuseCandidates(fused);

    // 4. 去重: 与 baseline 比较
    for (auto& f : fused)
    {
        bool isDuplicate = false;
        for (auto& b : baseline)
        {
            if (_ellipseIoU(f, b) >= cfg.fusionIoU)
            {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate)
            detEllipses.push_back(f);  // detEllipses 已有 baseline
    }
}

// 坐标回映射
void FLED::_remapCandidates(int scale,
                            std::vector<cv::RotatedRect>& detections,
                            std::vector<ScaleCandidate>& out)
{
    double invScale = 1.0 / scale;
    for (auto& d : detections)
    {
        ScaleCandidate c;
        c.scale = scale;
        c.ellipse.center.x = d.center.x * invScale;     // col
        c.ellipse.center.y = d.center.y * invScale;     // row
        c.ellipse.size.width  = d.size.width  * invScale;
        c.ellipse.size.height = d.size.height * invScale;
        c.ellipse.angle = d.angle;                       // 不变
        out.push_back(c);
    }
}

// IoU 计算 (复用 EllipseNonMaximumSuppression 的思路)
// 已有实现位于 EllipseNonMaximumSuppression.cpp，直接调用即可
double FLED::_ellipseIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) const
{
    // 可直接调用 EllipseNonMaximumSuppression 中的 IoU 函数
    // 伪代码: return computeEllipseIoU(a, b);
    // 当前 Python 实现见 ALGORITHM.md 引用的 overlap() 函数
}

// 构建 IoU 邻接图
void FLED::_buildIoUGraph(const std::vector<ScaleCandidate>& candidates,
                          std::vector<std::vector<int>>& graph) const
{
    int N = (int)candidates.size();
    graph.assign(N, std::vector<int>());
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            if (_ellipseIoU(candidates[i].ellipse, candidates[j].ellipse) >= _ms_cfg.fusionIoU)
            {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
}

// BFS 连通分量
void FLED::_findClusters(const std::vector<std::vector<int>>& graph,
                         int N, std::vector<std::vector<int>>& clusters) const
{
    std::vector<bool> visited(N, false);
    for (int i = 0; i < N; i++)
    {
        if (visited[i]) continue;
        std::vector<int> comp;
        std::queue<int> q;
        q.push(i); visited[i] = true;
        while (!q.empty())
        {
            int v = q.front(); q.pop();
            comp.push_back(v);
            for (int u : graph[v])
                if (!visited[u]) { visited[u] = true; q.push(u); }
        }
        clusters.push_back(comp);
    }
}

// 簇内代表选择 (最高平均 IoU, 平局时优先高尺度)
int FLED::_selectRepresentative(const std::vector<int>& cluster,
                                const std::vector<ScaleCandidate>& candidates,
                                const std::vector<std::vector<int>>& graph) const
{
    int best = cluster[0];
    double bestAvg = 0;

    for (int idx : cluster)
    {
        double sum = 0;
        int count = 0;
        for (int j : cluster)
        {
            if (j != idx)
            {
                sum += _ellipseIoU(candidates[idx].ellipse, candidates[j].ellipse);
                count++;
            }
        }
        double avg = (count > 0) ? sum / count : 1.0;

        if (avg > bestAvg ||
            (std::abs(avg - bestAvg) < 1e-9 &&
             candidates[idx].scale > candidates[best].scale))
        {
            best = idx;
            bestAvg = avg;
        }
    }
    return best;
}

// 跨尺度严格共识融合
void FLED::_fuseCandidates(std::vector<cv::RotatedRect>& fused)
{
    int N = (int)_ms_candidates.size();
    if (N == 0) return;

    std::vector<std::vector<int>> graph;
    _buildIoUGraph(_ms_candidates, graph);

    std::vector<std::vector<int>> clusters;
    _findClusters(graph, N, clusters);

    for (auto& cluster : clusters)
    {
        // 收集分支标签
        std::set<std::string> branches;
        bool has2x = false, has3x = false;
        for (int idx : cluster)
        {
            std::string label = "s" + std::to_string(_ms_candidates[idx].scale)
                              + "_t" + std::to_string((int)(_ms_candidates[idx].tVal * 100));
            branches.insert(label);
            if (_ms_candidates[idx].scale == 2) has2x = true;
            if (_ms_candidates[idx].scale == 3) has3x = true;
        }

        // 严格共识: ≥N_min 分支 + 跨尺度
        if (branches.size() >= (size_t)_ms_cfg.minBranches && has2x && has3x)
        {
            int rep = _selectRepresentative(cluster, _ms_candidates, graph);
            fused.push_back(_ms_candidates[rep].ellipse);
        }
    }
}
```

### 3. `src/main.cpp` — CLI 入口 (+30 行)

在 `DemoOptions` 结构体中新增：

```cpp
bool enableMultiScale = false;
MultiScaleConfig msConfig;
```

在 `parseArgs` 中新增：

```cpp
else if (arg == "--multi-scale" && idx + 1 < argc)
{
    options.enableMultiScale = true;
    // 解析逗号分隔的尺度列表 "2,3"
    std::string scales = argv[++idx];
    // parse comma-separated ints into options.msConfig.scales
}
else if (arg == "--multi-fusion" && idx + 1 < argc)
    options.msConfig.fusionMode = std::stoi(argv[++idx]);
else if (arg == "--multi-iou" && idx + 1 < argc)
    options.msConfig.fusionIoU = std::stod(argv[++idx]);
else if (arg == "--multi-min-branches" && idx + 1 < argc)
    options.msConfig.minBranches = std::stoi(argv[++idx]);
```

在 `processImage` 中（设置参数后，调用 `run_FLED` 前）：

```cpp
if (options.enableMultiScale)
    aamed.run_FLED_MultiScale(imgGray, options.msConfig);
else
    aamed.run_FLED(imgGray);
```

### 4. `src/FLED_Initialization.cpp` — 新增初始化 (可选)

如果在构造函数中需要初始化 `_ms_cfg` 默认值：

```cpp
// FLED 构造函数中添加:
_ms_cfg = MultiScaleConfig();  // 默认值已由结构体定义
```

## 不改的文件清单 (确认零影响)

- `src/Segmentation.cpp` — FSA 弧分割不变
- `src/Group.cpp` — 弧分组/邻接矩阵不变
- `src/Validation.cpp` — 椭圆验证不变
- `src/Contours.cpp` — 轮廓追踪不变
- `src/FLED_PrivateFunctions.cpp` — 辅助函数不变
- `src/FLED_Export.cpp` — 调试导出不变
- `src/FLED_drawAndWriteFunctions.cpp` — 写入 fled.txt 不变
- `src/adaptApproxPolyDP.cpp` — DP 近似不变
- `src/LinkMatrix.cpp` — 邻接矩阵不变
- `src/Node_FC.cpp` — 节点构造不变
- `src/EllipseNonMaximumSuppression.cpp` — NMS 不变 (但 IoU 函数可复用)
- `src/SCCDeferred.cpp` — 延迟选择不变

## 验证方法

1. 编译后运行 baseline 对比：
```bash
aamed_demo --input 002_0038.jpg --output-dir out_baseline --T-val 0.77
aamed_demo --input 002_0038.jpg --output-dir out_multi --T-val 0.77 --multi-scale 2,3 --multi-fusion 3
diff out_baseline/002_0038.jpg.fled.txt out_multi/002_0038.jpg.fled.txt
# 预期: 多尺度版本有 ≥ baseline 的检测数
```

2. 完整数据集评估 (与 Python 参考结果对比):
```bash
# C++ 多尺度
aamed_demo --input-list imagelist.txt --output-dir out_cpp_multi \
           --T-val 0.77 --multi-scale 2,3 --multi-fusion 3 --quiet
aamed_eval --results-dir out_cpp_multi --dataset-root datasets/prasad ...
# 预期: Precision≈0.655, Recall≈0.458, FMeasure≈0.539
```

## 参考 Python 实现

`reference/run_scc_small_fpn_eval.py` 中的关键函数与 C++ 方法的对应关系:

| Python 函数 | C++ 方法 | 行号 |
|---|---|---|
| `run_aamed_batch()` | `processImage()` (main.cpp) | ~260 |
| `remap_detections()` | `_remapCandidates()` | ~300 |
| `filter_small_candidates()` | 2.4 步的 r < maxRadius 判断 | ~305 |
| `build_iou_graph()` | `_buildIoUGraph()` | ~370 |
| `find_clusters()` | `_findClusters()` | ~380 |
| `select_representative()` | `_selectRepresentative()` | ~425 |
| `_fusion_strict()` | `_fuseCandidates()` | ~546 |
| `overlap()` (IoU) | `_ellipseIoU()` | 已有 |
