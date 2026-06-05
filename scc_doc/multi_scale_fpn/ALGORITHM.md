# 多尺度跨共识融合算法 — 精确描述

## 输入

| 符号 | 含义 |
|---|---|
| `img` | 灰度图像，尺寸 H×W |
| `S = {2, 3}` | 上采样尺度集合 |
| `T = {0.77, 0.72}` | 验证阈值集合 |
| `τ_iou = 0.8` | 融合 IoU 阈值 |
| `r_max = 20` | 小椭圆候选半径上限（回映射后半短轴几何均值） |
| `N_min = 3` | 严格共识最少支持分支数 |

## 输出

`fled.txt` 格式的椭圆检测列表，包含 baseline 检测 + 多尺度融合新增候选。

---

## 第一步：Baseline 检测

```
输入: img (1×)
过程: AAMED(img, T_val=0.77)
输出: B = {(cx, cy, a, b, θ)₁, ...}   // 1-indexed, semi-axes, radians
```

## 第二步：多尺度分支检测

对每一对 `(s, t) ∈ S × T`：

```
2.1 上采样:
    img_s = resize(img, (W*s, H*s), INTER_CUBIC)

2.2 运行 AAMED:
    D_raw = AAMED(img_s, T_val=t)

2.3 坐标回映射 (对 D_raw 中每个椭圆 e):
    cx' = (e.cx - 1) / s + 1
    cy' = (e.cy - 1) / s + 1
    a'  = e.a / s
    b'  = e.b / s
    θ'  = e.θ                         // 角度不变

2.4 小椭圆过滤:
    保留满足 sqrt(a' × b') < r_max 的候选

输出: C_{s,t} = { 回映射后的小椭圆候选 }
```

共产生 4 组候选：`C_{2,0.77}`, `C_{2,0.72}`, `C_{3,0.77}`, `C_{3,0.72}`

## 第三步：跨尺度严格共识融合 (cross_scale_strict)

```
3.1 构建候选池:
    all = []
    for each C_{s,t}:
        for each 椭圆 e in C_{s,t}:
            all.append( (branch_label, e) )
            // branch_label 编码尺度和阈值, 如 "s2_t77", "s2_t72", "s3_t77", "s3_t72"

3.2 构建 IoU 邻接图:
    N = len(all)
    graph = 邻接表, 大小 N
    for i in 0..N-1:
        for j in i+1..N-1:
            if ellipse_iou(all[i].e, all[j].e) ≥ τ_iou:
                graph[i].append(j)
                graph[j].append(i)

3.3 连通分量聚类 (BFS):
    visited[N] = {false}
    clusters = []
    for i in 0..N-1:
        if visited[i]: continue
        component = BFS(i, graph, visited)
        clusters.append(component)

3.4 对每个簇进行共识检查:
    fused = []
    for each cluster in clusters:
        // 收集该簇覆盖的分支标签
        branches = { all[idx].branch_label for idx in cluster }

        // 判断是否跨 2× 和 3×
        has_2x = "s2_t77" in branches or "s2_t72" in branches
        has_3x = "s3_t77" in branches or "s3_t72" in branches

        // 严格共识: ≥3 个分支 且 覆盖两种尺度
        if len(branches) ≥ N_min and has_2x and has_3x:
            // 选择簇内代表 (见 3.5)
            rep_idx = select_representative(cluster, all, graph)
            fused.append(all[rep_idx].e)

3.5 簇内代表选择:
    select_representative(cluster, all, graph):
        best = cluster[0]
        best_avg_iou = avg_iou_with_others(best, cluster, all, graph)
        best_scale = scale_of(all[best].branch_label)

        for idx in cluster[1:]:
            avg = avg_iou_with_others(idx, cluster, all, graph)
            s   = scale_of(all[idx].branch_label)
            if avg > best_avg_iou or (|avg - best_avg_iou| < 1e-9 and s > best_scale):
                best = idx; best_avg_iou = avg; best_scale = s

        return best

    avg_iou_with_others(idx, cluster, all, graph):
        对 cluster 中所有 j ≠ idx 的候选计算 ellipse_iou(all[idx].e, all[j].e)
        返回均值

    scale_of(branch_label):
        "s2_*" → 2,  "s3_*" → 3
```

## 第四步：去重 + 合并

```
4.1 与 baseline 去重:
    for each 候选 e in fused:
        if 存在 b in B 使得 ellipse_iou(e, b) ≥ τ_iou:
            跳过 e  // 此为 baseline 已有的检测

4.2 最终输出:
    F = B ∪ fused  (保留 baseline 全部检测，追加新候选)
```

## 第五步 (可选)：与 baseline 内部去重

如果 fused 内部有两个候选高度重叠，只保留得分最高的：

```
5.1 在 fused 内部再跑一次 NMS:
    fused_filtered = NMS(fused, τ_iou=0.8)
    // 按 cross_branch_avg_iou 排序，重叠者只留第一个
```

## 关键参数表

| 参数 | 值 | 位置 | 敏感性 |
|---|---|---|---|
| 上采样尺度 | 2, 3 | 第二步 | 低。加 4× 可提升 recall 但时间翻倍 |
| 验证阈值 | 0.77, 0.72 | 第二步 | 中。0.72 比 0.77 多约 30% 候选 |
| 回映射半径上限 | 20px | 2.4 | 低。设太小会漏候选，太大引入噪声 |
| 融合 IoU | 0.8 | 3.2 | 中。0.7-0.85 间差异不大 |
| 最少分支数 | 3 | 3.4 | 高。2→recall 涨但 precision 跌；4→precision 涨但 recall 跌 |
| 插值方法 | INTER_CUBIC | 2.1 | 低。LINEAR 略快，精度差异可忽略 |

## ellipse_iou 计算

```
ellipse_iou(e1, e2):
    // 数值积分法: 从两个椭圆的外接矩形取 min/max y 范围
    // 步长 0.2px, 对每个 y 求 x 交集的长度
    // 返回: intersection_area / union_area

    详见 reference/run_scc_small_fpn_eval.py 中 overlap() 函数
    (对应 AAMED 源码中 EllipseNonMaximumSuppression.cpp 的 IoU 实现)
```
