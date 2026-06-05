"""Small-ellipse FPN-style multi-scale validation experiment.

Stage 1: Quick screening on 40 diagnostic images.
Stage 2: Full validation on 198 Prasad images (only if Stage 1 passes).

Multi-scale pipeline:
  original -> INTER_CUBIC upscale 2x/3x -> AAMED -> remap coords back
  -> filter small ellipses -> cross-scale consistency fusion -> evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "prasad"
IMAGES = DATASET / "images"
GT_DIR = DATASET / "gt"
IMAGE_NAMES_PATH = DATASET / "imagenames.txt"
DEMO = ROOT / "build" / "bin" / "aamed_demo.exe"
EVAL_EXE = ROOT / "build" / "bin" / "aamed_eval.exe"
OUTPUT = ROOT / "output" / "scc_small_fpn"
DOC_DIR = ROOT / "scc_doc" / "02_experiments" / "06_small_ellipse_pyramid"

# Baseline reference directory (current branch, T=0.77)
BASELINE_DIR = ROOT / "output" / "prasad_migrated_current_branch"

SEED = 20260605
BASELINE_PRECISION = 0.779070
BASELINE_RECALL = 0.402575
BASELINE_FMEASURE = 0.530843

SMALL_RADIUS_THRESHOLD = 15.0       # sqrt(a*b) < 15px = "small ellipse"
REMAP_RADIUS_THRESHOLD = 20.0       # keep remapped candidates with sqrt(a*b) < 20
IOU_THRESHOLD = 0.8

# Module-level experiment parameters (set from CLI args)
_scale_mode: int = 0
_fsa_relax: int = 0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in 1-indexed coordinates with semi-axes and radians."""
    cx: float
    cy: float
    a: float   # semi-axis
    b: float   # semi-axis
    theta: float  # radians

    @property
    def radius(self) -> float:
        return math.sqrt(self.a * self.b)

    @property
    def is_small(self) -> bool:
        return self.radius < SMALL_RADIUS_THRESHOLD


@dataclass
class BranchResult:
    """Per-image result from one scale/threshold branch."""
    name: str
    scale: int
    t_val: float
    image_name: str
    raw_detections: list[Ellipse] = field(default_factory=list)
    remapped_detections: list[Ellipse] = field(default_factory=list)
    small_candidates: list[Ellipse] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class FusionVariant:
    """Output of a fusion rule."""
    name: str
    detections: dict[str, list[Ellipse]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gt(path: Path) -> list[Ellipse]:
    """Read Prasad-format GT file.

    Prasad GT format: cx cy a b theta (1-indexed centers, semi-axes, radians).
    Note: a/b are SEMI-AXES (radii), NOT full axes (diameters).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines[1:]:
        values = [float(v) for v in line.split()]
        if len(values) >= 5:
            result.append(Ellipse(
                cx=values[0], cy=values[1],
                a=values[2], b=values[3],
                theta=values[4],
            ))
    return result


def read_fled(path: Path) -> tuple[float, list[Ellipse]]:
    """Read AAMED fled.txt output.

    Returns (elapsed_ms, list_of_ellipses_in_1_indexed_semi_axes_radians).
    """
    if not path.exists():
        return 0.0, []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    elapsed = float(lines[0]) if lines else 0.0
    detections = []
    for line in lines[1:]:
        values = [float(v) for v in line.split()]
        if len(values) < 6 or int(values[0]) == 2:
            continue
        detections.append(Ellipse(
            cx=values[2] + 1.0,
            cy=values[1] + 1.0,
            a=values[3] / 2.0,
            b=values[4] / 2.0,
            theta=-values[5] / 180.0 * math.pi,
        ))
    return elapsed, detections


def write_fled(path: Path, ellipses: list[Ellipse], elapsed_ms: float = 0.0) -> None:
    """Write ellipses to AAMED fled.txt format.

    Input: 1-indexed, semi-axes, radians -> output: 0-indexed, full axes, degrees.
    """
    lines = [f"{elapsed_ms:.4f}"]
    for e in ellipses:
        lines.append(
            f"1 {e.cy - 1.0:.6f} {e.cx - 1.0:.6f} "
            f"{e.a * 2.0:.6f} {e.b * 2.0:.6f} "
            f"{-e.theta * 180.0 / math.pi:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Ellipse geometry (IoU via numerical integration)
# ---------------------------------------------------------------------------


def ellipse_equation(ellipse: Ellipse) -> tuple[float, ...]:
    xc, yc, a, b, theta = ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.theta
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    sin_2t = math.sin(2.0 * theta)
    aa_inv = 1.0 / (a * a)
    bb_inv = 1.0 / (b * b)
    params = [
        cos_t * cos_t * aa_inv + sin_t * sin_t * bb_inv,
        -0.5 * sin_2t * (bb_inv - aa_inv),
        cos_t * cos_t * bb_inv + sin_t * sin_t * aa_inv,
        (-xc * sin_t * sin_t + yc * sin_2t / 2.0) * bb_inv
        - (xc * cos_t * cos_t + yc * sin_2t / 2.0) * aa_inv,
        (-yc * cos_t * cos_t + xc * sin_2t / 2.0) * bb_inv
        - (yc * sin_t * sin_t + xc * sin_2t / 2.0) * aa_inv,
        ((xc * cos_t + yc * sin_t) / a) ** 2
        + ((yc * cos_t - xc * sin_t) / b) ** 2
        - 1.0,
    ]
    scale = 1.0 / math.sqrt(abs(params[0] * params[2] - params[1] * params[1]))
    return tuple(v * scale for v in params)


def range_at_y(equation: tuple[float, ...], y: float) -> tuple[float, float] | None:
    a, b, c, d, e, f = equation
    delta = (b * y + d) ** 2 - a * (c * y * y + 2.0 * e * y + f)
    if delta < 0.0:
        return None
    root = math.sqrt(delta)
    x1 = (-(b * y + d) - root) / a
    x2 = (-(b * y + d) + root) / a
    return (min(x1, x2), max(x1, x2))


def overlap(lhs: Ellipse, rhs: Ellipse) -> float:
    """Compute IoU between two ellipses via numerical integration."""
    lhs_eq = ellipse_equation(lhs)
    rhs_eq = ellipse_equation(rhs)
    y_min = math.floor(max(lhs.cy - max(lhs.a, lhs.b), rhs.cy - max(rhs.a, rhs.b)))
    y_max = math.ceil(min(lhs.cy + max(lhs.a, lhs.b), rhs.cy + max(rhs.a, rhs.b)))
    if y_min >= y_max:
        return 0.0
    step = 0.2
    intersection_sum = 0.0
    y = float(y_min)
    while y <= y_max + 1e-6:
        lhs_range = range_at_y(lhs_eq, y)
        rhs_range = range_at_y(rhs_eq, y)
        if lhs_range is not None and rhs_range is not None:
            x_min = max(lhs_range[0], rhs_range[0])
            x_max = min(lhs_range[1], rhs_range[1])
            if x_min < x_max:
                intersection_sum += x_max - x_min
        y += step
    intersection = intersection_sum * step
    union = math.pi * lhs.a * lhs.b + math.pi * rhs.a * rhs.b - intersection
    return intersection / union if union > 0.0 else 0.0


def match_any(ellipse: Ellipse, candidates: list[Ellipse], threshold: float = IOU_THRESHOLD) -> bool:
    return any(overlap(ellipse, c) > threshold for c in candidates)


# ---------------------------------------------------------------------------
# Multi-scale detection
# ---------------------------------------------------------------------------


def upscale_image(image_path: Path, scale: int) -> np.ndarray | None:
    """Upscale image by `scale` using INTER_CUBIC. Returns BGR array."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def run_aamed_batch(image_names: list[str], output_dir: Path,
                    t_val: float, scale_factor: float = 1.0,
                    scale_mode: int = 0, fsa_relax: int = 0,
                    timeout: int = 600) -> dict[str, tuple[float, list[Ellipse]]]:
    """Run AAMED in batch mode on multiple images (single subprocess call).

    Uses --input-list and --scale-factor for JIT upscaling.
    Returns {image_name: (elapsed_ms, detections)}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write image list file with absolute paths
    list_path = output_dir / "_image_list.txt"
    lines = [str((IMAGES / name).resolve()) for name in image_names]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [str(DEMO), "--input-list", str(list_path), "--output-dir", str(output_dir),
           "--T-val", str(t_val)]
    if abs(scale_factor - 1.0) > 1e-6:
        cmd.extend(["--scale-factor", str(scale_factor)])
    if scale_mode > 0:
        cmd.extend(["--scale-mode", str(scale_mode)])
    if fsa_relax > 0:
        cmd.extend(["--fsa-relax", str(fsa_relax)])
    cmd.append("--quiet")

    results: dict[str, tuple[float, list[Ellipse]]] = {}
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"  Batch failed (rc={proc.returncode}): {proc.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        print(f"  Batch timeout after {timeout}s")

    # Read results for all images (even if batch had some failures)
    for name in image_names:
        fled_path = output_dir / f"{name}.fled.txt"
        if fled_path.exists():
            results[name] = read_fled(fled_path)
        else:
            results[name] = (0.0, [])

    # Clean up list file
    list_path.unlink(missing_ok=True)
    return results


def remap_detections(detections: list[Ellipse], scale: int) -> list[Ellipse]:
    """Map detections from upscaled coordinates back to original scale."""
    result = []
    for det in detections:
        result.append(Ellipse(
            cx=(det.cx - 1.0) / scale + 1.0,
            cy=(det.cy - 1.0) / scale + 1.0,
            a=det.a / scale,
            b=det.b / scale,
            theta=det.theta,
        ))
    return result


def filter_small_candidates(detections: list[Ellipse], max_radius: float = REMAP_RADIUS_THRESHOLD) -> list[Ellipse]:
    """Keep only candidates with geometric mean radius < max_radius."""
    return [d for d in detections if math.sqrt(d.a * d.b) < max_radius]


# ---------------------------------------------------------------------------
# Scale branch definitions
# ---------------------------------------------------------------------------

SCALE_BRANCHES = [
    ("scale2_t77", 2, 0.77),
    ("scale2_t72", 2, 0.72),
    ("scale3_t77", 3, 0.77),
    ("scale3_t72", 3, 0.72),
]


def detect_scale_branch(image_name: str, branch_name: str, scale: int, t_val: float,
                        branch_dir: Path, timeout: int) -> BranchResult:
    """Run one scale/threshold branch for one image."""
    image_path = IMAGES / image_name
    result = BranchResult(name=branch_name, scale=scale, t_val=t_val, image_name=image_name)

    # Read original image to get size
    original = cv2.imread(str(image_path))
    if original is None:
        return result

    # Upscale
    upscaled = upscale_image(image_path, scale)
    if upscaled is None:
        return result

    # Run AAMED on upscaled image
    stem = Path(image_name).stem
    work_dir = branch_dir / "raw"
    elapsed, raw_detections = run_aamed(upscaled, work_dir, f"{stem}_s{scale}_t{int(t_val*100):02d}",
                                        t_val, timeout)
    result.raw_detections = raw_detections
    result.elapsed_ms = elapsed

    # Remap back to original coordinates
    result.remapped_detections = remap_detections(raw_detections, scale)

    # Filter small candidates
    result.small_candidates = filter_small_candidates(result.remapped_detections)

    # Write remapped result
    remapped_dir = branch_dir / "remapped"
    remapped_dir.mkdir(parents=True, exist_ok=True)
    write_fled(remapped_dir / f"{image_name}.fled.txt", result.remapped_detections, elapsed)

    # Write small-candidate-only result
    small_dir = branch_dir / "small_only"
    small_dir.mkdir(parents=True, exist_ok=True)
    write_fled(small_dir / f"{image_name}.fled.txt", result.small_candidates, elapsed)

    return result


# ---------------------------------------------------------------------------
# Cross-scale fusion
# ---------------------------------------------------------------------------


def build_iou_graph(candidates: list[tuple[str, Ellipse]]) -> dict[int, list[int]]:
    """Build adjacency list of IoU >= threshold connections.

    Each candidate is identified by its index. Returns {idx: [neighbor_indices]}.
    """
    n = len(candidates)
    graph: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if overlap(candidates[i][1], candidates[j][1]) >= IOU_THRESHOLD:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def find_clusters(graph: dict[int, list[int]], n: int) -> list[list[int]]:
    """Find connected components in the IoU graph."""
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        stack = [i]
        visited[i] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph.get(node, []):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        clusters.append(component)
    return clusters


def cross_branch_avg_iou(idx: int, cluster: list[int],
                         candidates: list[tuple[str, Ellipse]],
                         graph: dict[int, list[int]]) -> float:
    """Average IoU of candidate `idx` with all OTHER candidates in the cluster."""
    others = [j for j in cluster if j != idx]
    if not others:
        return 1.0
    ious = []
    for j in others:
        ious.append(overlap(candidates[idx][1], candidates[j][1]))
    return sum(ious) / len(ious)


def select_representative(cluster: list[int],
                          candidates: list[tuple[str, Ellipse]],
                          graph: dict[int, list[int]]) -> int:
    """Select best candidate per cluster: highest cross-branch avg IoU, tie → higher scale.

    candidate = (branch_name, ellipse)
    """
    def scale_order(branch_name: str) -> int:
        # "scale2_..." -> 2, "scale3_..." -> 3; generic labels -> 0
        if branch_name.startswith("scale"):
            try:
                return int(branch_name[5])
            except (ValueError, IndexError):
                pass
        return 0

    best_idx = cluster[0]
    best_avg = cross_branch_avg_iou(best_idx, cluster, candidates, graph)
    best_scale = scale_order(candidates[best_idx][0])

    for idx in cluster[1:]:
        avg = cross_branch_avg_iou(idx, cluster, candidates, graph)
        s = scale_order(candidates[idx][0])
        if avg > best_avg or (abs(avg - best_avg) < 1e-9 and s > best_scale):
            best_idx = idx
            best_avg = avg
            best_scale = s

    return best_idx


def fusion_union(baseline: list[Ellipse], scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Union: baseline + all scale candidates not duplicating baseline."""
    result = list(baseline)
    for branch_name, candidates in scale_candidates.items():
        for c in candidates:
            if not match_any(c, result):
                result.append(c)
    return result


def fusion_stable_scale2(baseline: list[Ellipse],
                         scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Candidates that appear in BOTH scale2_t77 AND scale2_t72 with IoU >= 0.8."""
    t77 = scale_candidates.get("scale2_t77", [])
    t72 = scale_candidates.get("scale2_t72", [])
    stable = _intersect_branches([t77, t72])
    result = list(baseline)
    for c in stable:
        if not match_any(c, result):
            result.append(c)
    return result


def fusion_cross_scale(baseline: list[Ellipse],
                       scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Candidates appearing in at least one 2x branch AND one 3x branch."""
    scale2 = scale_candidates.get("scale2_t77", []) + scale_candidates.get("scale2_t72", [])
    scale3 = scale_candidates.get("scale3_t77", []) + scale_candidates.get("scale3_t72", [])
    return _fusion_cross_scale_min(baseline, scale_candidates, scale2, scale3, min_count=1)


def fusion_cross_scale_strict(baseline: list[Ellipse],
                              scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Candidates supported by >= 3 branches covering both 2x and 3x."""
    all_branches = [scale_candidates.get(name, []) for name, _, _ in SCALE_BRANCHES]
    return _fusion_strict(baseline, scale_candidates, all_branches, min_branches=3)


def _intersect_branches(branch_lists: list[list[Ellipse]]) -> list[Ellipse]:
    """Find candidates present in ALL branch lists (pairwise IoU >= 0.8 matching)."""
    if not branch_lists or any(len(b) == 0 for b in branch_lists):
        return []
    # Build a combined list with branch labels
    labeled = []
    for bi, branch in enumerate(branch_lists):
        for ellipse in branch:
            labeled.append((f"branch_{bi}", ellipse))
    if not labeled:
        return []
    graph = build_iou_graph(labeled)
    clusters = find_clusters(graph, len(labeled))
    # A cluster is in the intersection if it has at least one member from each branch
    result = []
    for cluster in clusters:
        branches_present = {labeled[idx][0] for idx in cluster}
        if len(branches_present) == len(branch_lists):
            rep_idx = select_representative(cluster, labeled, graph)
            result.append(labeled[rep_idx][1])
    return result


def _fusion_cross_scale_min(baseline: list[Ellipse],
                            scale_candidates: dict[str, list[Ellipse]],
                            scale2_pool: list[Ellipse],
                            scale3_pool: list[Ellipse],
                            min_count: int = 1) -> list[Ellipse]:
    """Candidates with support from both scale groups."""
    # Label candidates
    labeled = []
    for e in scale2_pool:
        labeled.append(("scale2", e))
    for e in scale3_pool:
        labeled.append(("scale3", e))
    if not labeled:
        return list(baseline)

    graph = build_iou_graph(labeled)
    clusters = find_clusters(graph, len(labeled))

    result = list(baseline)
    for cluster in clusters:
        scales_present = {labeled[idx][0] for idx in cluster}
        if "scale2" in scales_present and "scale3" in scales_present:
            rep_idx = select_representative(cluster, labeled, graph)
            c = labeled[rep_idx][1]
            if not match_any(c, result):
                result.append(c)
    return result


def _fusion_strict(baseline: list[Ellipse],
                   scale_candidates: dict[str, list[Ellipse]],
                   all_branches: list[list[Ellipse]],
                   min_branches: int = 3) -> list[Ellipse]:
    """Candidates supported by >= min_branches, covering both 2x and 3x."""
    labeled = []
    for bi, branch in enumerate(all_branches):
        name = f"b{bi}"
        for e in branch:
            labeled.append((name, e))
    if not labeled:
        return list(baseline)

    graph = build_iou_graph(labeled)
    clusters = find_clusters(graph, len(labeled))

    result = list(baseline)
    for cluster in clusters:
        branches_present = {labeled[idx][0] for idx in cluster}
        # Determine which scale groups are covered
        branch_names = ["b0", "b1", "b2", "b3"]  # s2_t77, s2_t72, s3_t77, s3_t72
        has_scale2 = ("b0" in branches_present) or ("b1" in branches_present)
        has_scale3 = ("b2" in branches_present) or ("b3" in branches_present)
        if len(branches_present) >= min_branches and has_scale2 and has_scale3:
            rep_idx = select_representative(cluster, labeled, graph)
            c = labeled[rep_idx][1]
            if not match_any(c, result):
                result.append(c)
    return result


# ---------------------------------------------------------------------------
# Stage 3: SNIP-style scale range filtering + Adaptive IoU + FPN confidence
# ---------------------------------------------------------------------------


def snip_filter_candidates(scale_candidates: dict[str, list[Ellipse]]) -> dict[str, list[Ellipse]]:
    """SNIP-style per-scale radius limits.

    Each scale should only detect objects in its effective size range.
    - 2x branches: keep r < 25px (small-to-medium)
    - 3x branches: keep r < 18px (small only)
    Larger objects are better detected at 1x.
    """
    limits = {
        "scale2_t77": 25.0,
        "scale2_t72": 25.0,
        "scale3_t77": 18.0,
        "scale3_t72": 18.0,
    }
    result = {}
    for branch_name, candidates in scale_candidates.items():
        max_r = limits.get(branch_name, 999.0)
        result[branch_name] = [c for c in candidates if math.sqrt(c.a * c.b) < max_r]
    return result


def adaptive_iou_overlap(lhs: Ellipse, rhs: Ellipse) -> float:
    """Compute IoU with adaptive threshold recommendation.

    Returns the raw IoU value; the caller decides the threshold based on size.
    """
    return overlap(lhs, rhs)


def adaptive_match_any(ellipse: Ellipse, candidates: list[Ellipse],
                       min_iou: float = 0.6) -> bool:
    """Match with size-adaptive IoU threshold.

    Small ellipses (r<15): IoU >= 0.60 (more lenient)
    Medium (15-30): IoU >= 0.70
    Large (30+): IoU >= 0.80 (strict)
    """
    r = math.sqrt(ellipse.a * ellipse.b)
    if r < 15.0:
        threshold = max(min_iou, 0.60)
    elif r < 30.0:
        threshold = max(min_iou, 0.70)
    else:
        threshold = max(min_iou, 0.80)
    return any(overlap(ellipse, c) > threshold for c in candidates)


def fusion_adaptive_iou(baseline: list[Ellipse],
                        scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Fusion with adaptive IoU: size-dependent matching thresholds."""
    result = list(baseline)
    # First apply SNIP filter
    filtered = snip_filter_candidates(scale_candidates)
    # Then add non-duplicate candidates with adaptive matching
    for branch_name, candidates in filtered.items():
        for c in candidates:
            if not adaptive_match_any(c, result):
                result.append(c)
    return result


def fusion_cross_scale_adaptive(baseline: list[Ellipse],
                                scale_candidates: dict[str, list[Ellipse]]) -> list[Ellipse]:
    """Cross-scale consensus with adaptive IoU and SNIP filtering.

    Combines: SNIP filter + adaptive IoU + cross-scale consistency + FPN confidence.
    """
    # Apply SNIP filter first
    filtered = snip_filter_candidates(scale_candidates)

    # Build pool: scale2 candidates + scale3 candidates
    scale2_pool = filtered.get("scale2_t77", []) + filtered.get("scale2_t72", [])
    scale3_pool = filtered.get("scale3_t77", []) + filtered.get("scale3_t72", [])

    # Label candidates for clustering
    labeled = []
    for e in scale2_pool:
        labeled.append(("scale2", e))
    for e in scale3_pool:
        labeled.append(("scale3", e))
    if not labeled:
        return list(baseline)

    graph = build_iou_graph(labeled)
    clusters = find_clusters(graph, len(labeled))

    result = list(baseline)
    for cluster in clusters:
        scales_present = {labeled[idx][0] for idx in cluster}
        # FPN-style: candidates appearing in both scale groups get priority
        has_both = "scale2" in scales_present and "scale3" in scales_present
        if has_both:
            rep_idx = select_representative(cluster, labeled, graph)
            c = labeled[rep_idx][1]
            if not adaptive_match_any(c, result):
                result.append(c)
        # Single-scale candidates only added if they're small enough
        elif "scale3" in scales_present:
            # 3x-only: only keep if very small (r < 12px)
            rep_idx = select_representative(cluster, labeled, graph)
            c = labeled[rep_idx][1]
            r = math.sqrt(c.a * c.b)
            if r < 12.0 and not adaptive_match_any(c, result):
                result.append(c)

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_detections(name: str, detections: list[Ellipse],
                        gt: list[Ellipse] | None = None) -> dict:
    """Compute per-image metrics."""
    if gt is None:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
    gt_match = [match_any(truth, detections) for truth in gt]
    det_match = [match_any(det, gt) for det in detections]
    num_true = sum(gt_match)
    num_det = len(detections)
    num_gt = len(gt)
    precision = num_true / num_det if num_det else 0.0
    recall = num_true / num_gt if num_gt else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image": name,
        "matched": num_true,
        "detected": num_det,
        "gt_count": num_gt,
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure,
        "gt_match": gt_match,
    }


def aggregate_metrics(per_image: list[dict]) -> dict:
    """Aggregate per-image metrics into global metrics."""
    total_matched = sum(d["matched"] for d in per_image)
    total_detected = sum(d["detected"] for d in per_image)
    total_gt = sum(d["gt_count"] for d in per_image)
    precision = total_matched / total_detected if total_detected else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": len(per_image),
        "positive_matches": total_matched,
        "detected_count": total_detected,
        "ground_truth_count": total_gt,
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure,
    }


def size_group_recall(names: list[str], variant_detections: dict[str, list[Ellipse]]) -> dict:
    """Compute recall by size group."""
    groups = {
        "radius_lt15": [],
        "radius_15_30": [],
        "radius_30_60": [],
        "radius_60_plus": [],
    }
    for name in names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        dets = variant_detections.get(name, [])
        for truth in gt:
            r = math.sqrt(truth.a * truth.b)
            is_match = match_any(truth, dets)
            if r < 15:
                groups["radius_lt15"].append(is_match)
            elif r < 30:
                groups["radius_15_30"].append(is_match)
            elif r < 60:
                groups["radius_30_60"].append(is_match)
            else:
                groups["radius_60_plus"].append(is_match)
    return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in groups.items()}


def run_aamed_eval(results_dir: Path, report_path: Path) -> dict:
    """Run the C++ aamed_eval.exe and parse its output."""
    proc = subprocess.run(
        [str(EVAL_EXE), "--dataset-root", str(DATASET),
         "--results-dir", str(results_dir), "--gt-prefix", "gt_",
         "--gt-format", "plain_rad", "--result-format", "aamed_fled",
         "--overlap", "0.8", "--report", str(report_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"aamed_eval failed: {proc.stderr}")
    metrics = {}
    for raw in report_path.read_text(encoding="utf-8").splitlines():
        if ":" in raw:
            k, v = raw.split(":", 1)
            try:
                metrics[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return metrics


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_compare(names: list[str],
                      baseline_dets: dict[str, list[Ellipse]],
                      variant_dets: dict[str, list[Ellipse]],
                      samples: int = 10_000) -> dict:
    """Image-level bootstrap of Recall and FMeasure differences."""
    # Pre-compute per-image counts
    baseline_counts = []
    variant_counts = []
    for name in names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        b_dets = baseline_dets.get(name, [])
        v_dets = variant_dets.get(name, [])
        b_match = sum(match_any(t, b_dets) for t in gt)
        v_match = sum(match_any(t, v_dets) for t in gt)
        baseline_counts.append((b_match, len(b_dets), len(gt)))
        variant_counts.append((v_match, len(v_dets), len(gt)))

    rng = np.random.default_rng(SEED)
    f1_deltas = np.empty(samples)
    recall_deltas = np.empty(samples)

    def compute_f1_recall(counts, indices):
        pos = sum(counts[i][0] for i in indices)
        det = sum(counts[i][1] for i in indices)
        gt = sum(counts[i][2] for i in indices)
        p = pos / det if det else 0.0
        r = pos / gt if gt else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return f, r

    for s in range(samples):
        indices = rng.integers(0, len(names), len(names))
        b_f1, b_recall = compute_f1_recall(baseline_counts, indices)
        v_f1, v_recall = compute_f1_recall(variant_counts, indices)
        f1_deltas[s] = v_f1 - b_f1
        recall_deltas[s] = v_recall - b_recall

    return {
        "f1_delta_ci95": [float(x) for x in np.quantile(f1_deltas, [0.025, 0.975])],
        "recall_delta_ci95": [float(x) for x in np.quantile(recall_deltas, [0.025, 0.975])],
        "f1_delta_mean": float(np.mean(f1_deltas)),
        "recall_delta_mean": float(np.mean(recall_deltas)),
    }


# ---------------------------------------------------------------------------
# Diagnostic image selection (Stage 1)
# ---------------------------------------------------------------------------


def select_diagnostic_images(baseline_dir: Path, all_names: list[str]) -> tuple[list[str], dict]:
    """Select 40 diagnostic images for Stage 1.

    Returns (selected_names, selection_metadata).
    """
    rng = random.Random(SEED)

    # Classify images
    unmatched_small = []   # images with baseline-unmatched small ellipses
    matched_small = []     # images with baseline-matched small ellipses
    no_small = []          # images without small ellipses

    for name in all_names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        _, detections = read_fled(baseline_dir / f"{name}.fled.txt")
        small_gts = [e for e in gt if e.is_small]
        if not small_gts:
            no_small.append(name)
            continue
        has_unmatched = any(not match_any(e, detections) for e in small_gts)
        has_matched = any(match_any(e, detections) for e in small_gts)
        if has_unmatched:
            unmatched_small.append(name)
        elif has_matched:
            matched_small.append(name)

    # Select
    selected = []
    selected.extend(rng.sample(unmatched_small, min(20, len(unmatched_small))))
    selected.extend(rng.sample(matched_small, min(10, len(matched_small))))
    selected.extend(rng.sample(no_small, min(10, len(no_small))))

    # If any category insufficient, fill from others
    while len(selected) < 40:
        remaining = [n for n in all_names if n not in selected]
        if not remaining:
            break
        selected.append(rng.choice(remaining))

    rng.shuffle(selected)

    meta = {
        "total_selected": len(selected),
        "unmatched_small_pool": len(unmatched_small),
        "matched_small_pool": len(matched_small),
        "no_small_pool": len(no_small),
        "selected_unmatched_small": len([n for n in selected if n in unmatched_small]),
        "selected_matched_small": len([n for n in selected if n in matched_small]),
        "selected_no_small": len([n for n in selected if n in no_small]),
    }
    return selected, meta


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_ellipse_cv(image: np.ndarray, ellipse: Ellipse, color: tuple[int, int, int],
                    thickness: int = 2) -> None:
    """Draw an Ellipse on an OpenCV image (BGR)."""
    center = (round(ellipse.cx - 1), round(ellipse.cy - 1))
    axes = (max(1, round(ellipse.a)), max(1, round(ellipse.b)))
    angle_deg = ellipse.theta * 180.0 / math.pi
    cv2.ellipse(image, center, axes, angle_deg, 0, 360, color, thickness)


def render_case(image_name: str, gt: list[Ellipse],
                baseline_dets: list[Ellipse],
                variant_dets: list[Ellipse],
                title: str, output_path: Path) -> None:
    """Render a case visualization: green=GT, red=baseline, blue=variant."""
    img = cv2.imread(str(IMAGES / image_name))
    if img is None:
        return
    for truth in gt:
        draw_ellipse_cv(img, truth, (0, 255, 0), 2)       # green
    for det in baseline_dets:
        draw_ellipse_cv(img, det, (0, 0, 255), 1)          # red
    for det in variant_dets:
        draw_ellipse_cv(img, det, (255, 0, 0), 2)          # blue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def generate_case_visualizations(names: list[str],
                                 baseline_dets: dict[str, list[Ellipse]],
                                 variant_dets: dict[str, list[Ellipse]],
                                 output_dir: Path,
                                 max_cases: int = 12) -> list[str]:
    """Generate visualization images for best and worst cases."""
    scored = []
    for name in names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        b_match = sum(match_any(t, baseline_dets.get(name, [])) for t in gt)
        v_match = sum(match_any(t, variant_dets.get(name, [])) for t in gt)
        delta = v_match - b_match
        # Also count small ellipse recovery
        small_gt = [t for t in gt if t.is_small]
        small_b = sum(match_any(t, baseline_dets.get(name, [])) for t in small_gt)
        small_v = sum(match_any(t, variant_dets.get(name, [])) for t in small_gt)
        small_delta = small_v - small_b
        scored.append((delta, small_delta, name))

    # Top improvers and decliners
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [s[2] for s in scored[:4]]   # top 4 improvers
    scored.sort(key=lambda x: (x[0], x[1]))
    selected += [s[2] for s in scored[:4] if s[2] not in selected]  # top 4 decliners
    # Middle cases
    mid = len(scored) // 2
    for s in scored[mid:mid + 4]:
        if s[2] not in selected:
            selected.append(s[2])

    cases = []
    case_dir = output_dir / "cases"
    for name in selected[:max_cases]:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        render_case(name, gt,
                    baseline_dets.get(name, []),
                    variant_dets.get(name, []),
                    name, case_dir / f"{name}.png")
        cases.append(str((case_dir / f"{name}.png").relative_to(ROOT)))

    return cases


# ---------------------------------------------------------------------------
# Main experiment orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-ellipse FPN multi-scale experiment")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="Experiment stage (1=screening, 2=full validation)")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if results exist")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build step")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per image per branch (seconds)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run on a tiny subset for testing")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Reuse existing baseline results")
    parser.add_argument("--scale-mode", type=int, default=0,
                        help="0=INTER_CUBIC, 1=Laplacian, 2=CLAHE")
    parser.add_argument("--fsa-relax", type=int, default=0,
                        help="0=standard FSA, 1=relaxed FSA")
    return parser.parse_args()


def ensure_build(skip: bool) -> None:
    if skip:
        return
    print("Building AAMED...")
    subprocess.run([str(ROOT / "build.bat")], cwd=ROOT, check=True)


def get_baseline_detections(names: list[str], force: bool, timeout: int) -> dict[str, list[Ellipse]]:
    """Run baseline detection (T=0.77, 1x scale) using batch mode."""
    baseline_out = OUTPUT / "fusion_variants" / "baseline"
    baseline_out.mkdir(parents=True, exist_ok=True)

    needs_proc = []
    for name in names:
        if not force and (baseline_out / f"{name}.fled.txt").exists():
            continue
        needs_proc.append(name)

    if needs_proc:
        print(f"  Running baseline on {len(needs_proc)} images (batch mode)...")
        batch_results = run_aamed_batch(needs_proc, baseline_out, 0.77, 1.0,
                                        fsa_relax=_fsa_relax, timeout=timeout)
        # Results already written to fled.txt by batch mode

    detections = {}
    for name in names:
        fled_path = baseline_out / f"{name}.fled.txt"
        if fled_path.exists():
            _, dets = read_fled(fled_path)
            detections[name] = dets
        else:
            detections[name] = []
    return detections


def run_all_branches(names: list[str], force: bool, timeout: int) -> dict[str, dict[str, list[Ellipse]]]:
    """Run all 4 scale/threshold branches using batch mode (single subprocess per branch).

    Returns {branch_name: {image_name: [small_candidates]}}.
    """
    branch_results: dict[str, dict[str, list[Ellipse]]] = {}
    branches_dir = OUTPUT / "scale_branches"

    for branch_name, scale, t_val in SCALE_BRANCHES:
        branch_dir = branches_dir / branch_name
        remapped_dir = branch_dir / "remapped"
        small_dir = branch_dir / "small_only"
        remapped_dir.mkdir(parents=True, exist_ok=True)
        small_dir.mkdir(parents=True, exist_ok=True)

        # Determine which images need processing
        needs_proc = []
        for name in names:
            if not force and (small_dir / f"{name}.fled.txt").exists():
                continue
            needs_proc.append(name)

        print(f"\n{'='*60}")
        print(f"Branch: {branch_name} (scale={scale}x, T={t_val})")
        if needs_proc:
            print(f"  Processing {len(needs_proc)} images (batch mode, --scale-factor {scale})...")
        else:
            print(f"  All {len(names)} images cached.")
        print(f"{'='*60}")

        started = time.time()

        if needs_proc:
            # Single batch subprocess call: AAMED upscales internally, remaps coords in output
            # The output fled.txt already has original-scale coordinates!
            batch_results = run_aamed_batch(needs_proc, remapped_dir, t_val, float(scale),
                                            scale_mode=_scale_mode, fsa_relax=_fsa_relax,
                                            timeout=timeout)

            # Filter small candidates from the (already remapped) results
            for name in needs_proc:
                elapsed, dets = batch_results.get(name, (0.0, []))
                # dets are already in original coordinates (remapped by C++ --scale-factor)
                small = filter_small_candidates(dets)
                write_fled(small_dir / f"{name}.fled.txt", small, elapsed)

        # Build result dict from cached/generated files
        branch_results[branch_name] = {}
        ok = fail = 0
        for name in names:
            small_path = small_dir / f"{name}.fled.txt"
            if small_path.exists():
                _, dets = read_fled(small_path)
                branch_results[branch_name][name] = dets
                ok += 1
            else:
                branch_results[branch_name][name] = []
                fail += 1

        print(f"  Done: ok={ok} fail={fail} {time.time()-started:.1f}s")

    return branch_results


def apply_fusion_variants(names: list[str],
                          baseline_dets: dict[str, list[Ellipse]],
                          branch_dets: dict[str, dict[str, list[Ellipse]]]) -> dict[str, dict[str, list[Ellipse]]]:
    """Apply all 6 fusion variants and write fled.txt outputs.

    Returns {variant_name: {image_name: [ellipses]}}.
    """
    fusion_dir = OUTPUT / "fusion_variants"
    variants = {}

    for i, name in enumerate(names, 1):
        bl = baseline_dets.get(name, [])
        sc = {bn: branch_dets[bn].get(name, []) for bn, _, _ in SCALE_BRANCHES}

        # Variant 1: baseline (already computed, just reference)
        # Variant 2: baseline + scale2_t77 union
        v2 = fusion_union(bl, {"scale2_t77": sc["scale2_t77"]})

        # Variant 3: baseline + all-scale union
        v3 = fusion_union(bl, sc)

        # Variant 4: baseline + scale2 stable
        v4 = fusion_stable_scale2(bl, sc)

        # Variant 5: baseline + cross-scale
        v5 = fusion_cross_scale(bl, sc)

        # Variant 6: baseline + cross-scale strict
        v6 = fusion_cross_scale_strict(bl, sc)

        # Stage 3 variants: SNIP + Adaptive IoU + FPN confidence
        # Variant 7: baseline + adaptive IoU fusion (all scales, SNIP-filtered)
        v7 = fusion_adaptive_iou(bl, sc)

        # Variant 8: baseline + cross-scale adaptive (SNIP + adaptive IoU + FPN)
        v8 = fusion_cross_scale_adaptive(bl, sc)

        variants[name] = {
            "baseline": bl,
            "union_scale2_t77": v2,
            "union_all_scale": v3,
            "stable_scale2": v4,
            "cross_scale": v5,
            "cross_scale_strict": v6,
            "adaptive_iou": v7,
            "cross_scale_adaptive": v8,
        }

    # Reorganize: {variant_name: {image_name: [ellipses]}}
    result: dict[str, dict[str, list[Ellipse]]] = defaultdict(dict)
    for name, var_dict in variants.items():
        for var_name, dets in var_dict.items():
            result[var_name][name] = dets

    # Write fled.txt for each variant
    for var_name in result:
        var_dir = fusion_dir / var_name
        var_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            write_fled(var_dir / f"{name}.fled.txt", result[var_name][name])

    return result


# ---------------------------------------------------------------------------
# Stage 1: Screening
# ---------------------------------------------------------------------------


def run_stage1(args: argparse.Namespace) -> dict:
    """Run Stage 1 on 40 diagnostic images."""
    global _scale_mode, _fsa_relax
    _scale_mode = args.scale_mode
    _fsa_relax = args.fsa_relax
    print("\n" + "=" * 70)
    print("STAGE 1: Quick Screening on 40 Diagnostic Images")
    print("=" * 70)

    all_names = read_names(IMAGE_NAMES_PATH)
    if args.smoke:
        all_names = all_names[:10]

    # Select diagnostic images
    diag_names, selection_meta = select_diagnostic_images(BASELINE_DIR, all_names)
    print(f"\nDiagnostic image selection:")
    for k, v in selection_meta.items():
        print(f"  {k}: {v}")

    # Save diagnostic image list
    stage1_dir = OUTPUT / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    (stage1_dir / "diagnostic_images.txt").write_text("\n".join(diag_names) + "\n")

    # Run baseline on diagnostic images
    print("\n--- Baseline ---")
    baseline_dets = get_baseline_detections(diag_names, args.force, args.timeout)

    # Run all 4 scale branches
    branch_dets = run_all_branches(diag_names, args.force, args.timeout)

    # Apply fusion
    print("\n--- Fusion ---")
    variant_dets = apply_fusion_variants(diag_names, baseline_dets, branch_dets)

    # Evaluate all variants
    print("\n--- Evaluation ---")
    variant_metrics = {}
    for var_name in ["baseline", "union_scale2_t77", "union_all_scale",
                      "stable_scale2", "cross_scale", "cross_scale_strict",
                      "adaptive_iou", "cross_scale_adaptive"]:
        per_image = [evaluate_detections(name, variant_dets[var_name].get(name, []))
                     for name in diag_names]
        agg = aggregate_metrics(per_image)
        variant_metrics[var_name] = agg

        # Size group recall
        size_recall = size_group_recall(diag_names, variant_dets[var_name])

        # Small ellipse recovery
        small_gt_total = 0
        small_recovered = 0
        for name in diag_names:
            gt = read_gt(GT_DIR / f"gt_{name}.txt")
            small_gts = [e for e in gt if e.is_small]
            small_gt_total += len(small_gts)
            bl_dets = baseline_dets.get(name, [])
            var_dets_list = variant_dets[var_name].get(name, [])
            for sg in small_gts:
                if not match_any(sg, bl_dets) and match_any(sg, var_dets_list):
                    small_recovered += 1

        # New candidate precision
        total_new = 0
        total_new_matched = 0
        for name in diag_names:
            bl_dets = baseline_dets.get(name, [])
            var_dets_list = variant_dets[var_name].get(name, [])
            gt = read_gt(GT_DIR / f"gt_{name}.txt")
            new = [d for d in var_dets_list if not match_any(d, bl_dets)]
            total_new += len(new)
            total_new_matched += sum(match_any(d, gt) for d in new)
        new_precision = total_new_matched / total_new if total_new else 0.0

        print(f"\n{var_name}:")
        print(f"  P={agg['precision']:.6f} R={agg['recall']:.6f} F={agg['fmeasure']:.6f}")
        print(f"  Small recall: {size_recall.get('radius_lt15', 0):.6f}")
        print(f"  Small recovered: {small_recovered}/{small_gt_total}")
        print(f"  New candidates: {total_new}, precision={new_precision:.6f}")

        agg["size_group_recall"] = size_recall
        agg["small_recovered"] = small_recovered
        agg["small_gt_total"] = small_gt_total
        agg["new_candidates"] = total_new
        agg["new_candidate_precision"] = new_precision
        variant_metrics[var_name] = agg

    # Check stage 1 gate criteria
    baseline_small_recall = variant_metrics["baseline"]["size_group_recall"]["radius_lt15"]
    best_union = variant_metrics["union_all_scale"]
    small_recall_delta = best_union["size_group_recall"]["radius_lt15"] - baseline_small_recall
    small_recovered_union = best_union["small_recovered"]
    # Find best non-GT fusion variant
    non_gt_variants = {k: v for k, v in variant_metrics.items() if k != "baseline"}
    best_new_precision = max(v["new_candidate_precision"] for v in non_gt_variants.values())

    gates = {
        "small_recovered_ge_10": small_recovered_union >= 10,
        "small_recall_delta_ge_005": small_recall_delta >= 0.05,
        "new_precision_ge_030": best_new_precision >= 0.30,
    }
    gates_pass = all(gates.values())

    stage1_summary = {
        "stage": 1,
        "gates_pass": gates_pass,
        "gates": gates,
        "gates_detail": {
            "small_recovered_union": small_recovered_union,
            "small_recall_delta": small_recall_delta,
            "best_new_precision": best_new_precision,
        },
        "selection_meta": selection_meta,
        "variant_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "gt_match"}
                            for k, v in variant_metrics.items()},
    }
    (stage1_dir / "stage1_summary.json").write_text(
        json.dumps(stage1_summary, indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"STAGE 1 GATES: {'ALL PASSED' if gates_pass else 'NOT PASSED'}")
    for gate, passed in gates.items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"  details: recovered={small_recovered_union}, "
          f"recall_delta={small_recall_delta:.4f}, "
          f"new_precision={best_new_precision:.4f}")

    if not gates_pass:
        print("\nStage 1 gates not met. Stopping. No AAMED core changes made.")
        print("To force Stage 2 anyway, use --stage 2 --force")

    return stage1_summary


# ---------------------------------------------------------------------------
# Stage 2: Full validation
# ---------------------------------------------------------------------------


def run_stage2(args: argparse.Namespace, stage1_summary: dict | None = None) -> dict:
    """Run Stage 2 on full 198-image Prasad dataset."""
    global _scale_mode, _fsa_relax
    _scale_mode = args.scale_mode
    _fsa_relax = args.fsa_relax
    print("\n" + "=" * 70)
    print("STAGE 2: Full Validation on 198 Prasad Images")
    print("=" * 70)

    if stage1_summary and not stage1_summary.get("gates_pass"):
        print("WARNING: Stage 1 gates were not met. Running Stage 2 anyway per --stage 2.")

    all_names = read_names(IMAGE_NAMES_PATH)
    if args.smoke:
        all_names = all_names[:20]

    stage2_dir = OUTPUT / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Only run variants that passed stage 1 criteria (or all if forced)
    passed_variants = None
    if stage1_summary and stage1_summary.get("gates_pass"):
        # Determine which variants to include based on stage 1 findings
        # Include all variants that showed potential
        passed_variants = ["baseline", "union_scale2_t77", "union_all_scale",
                           "stable_scale2", "cross_scale", "cross_scale_strict",
                           "adaptive_iou", "cross_scale_adaptive"]

    if passed_variants is None:
        passed_variants = ["baseline", "union_scale2_t77", "union_all_scale",
                           "stable_scale2", "cross_scale", "cross_scale_strict",
                           "adaptive_iou", "cross_scale_adaptive"]

    # Run baseline on full dataset
    print("\n--- Baseline (full dataset) ---")
    baseline_dets = get_baseline_detections(all_names, args.force, args.timeout)

    # Run all 4 scale branches
    branch_dets = run_all_branches(all_names, args.force, args.timeout)

    # Apply fusion
    print("\n--- Fusion ---")
    variant_dets = apply_fusion_variants(all_names, baseline_dets, branch_dets)

    # Evaluate via aamed_eval.exe for each variant
    print("\n--- C++ Evaluation ---")
    fusion_dir = OUTPUT / "fusion_variants"
    cpp_metrics = {}
    for var_name in passed_variants:
        var_dir = fusion_dir / var_name
        report_path = stage2_dir / f"{var_name}_eval.txt"
        try:
            metrics = run_aamed_eval(var_dir, report_path)
            cpp_metrics[var_name] = metrics
            print(f"  {var_name}: P={metrics['Precision']:.6f} R={metrics['Recall']:.6f} "
                  f"F={metrics['FMeasure']:.6f}")
        except Exception as e:
            print(f"  {var_name}: eval failed: {e}")

    # Python native evaluation (per-image detail)
    print("\n--- Python Evaluation ---")
    py_metrics = {}
    per_image_csv_rows = []
    for var_name in passed_variants:
        per_image = []
        for name in all_names:
            gt = read_gt(GT_DIR / f"gt_{name}.txt")
            dets = variant_dets[var_name].get(name, [])
            img_metrics = evaluate_detections(name, dets, gt)
            per_image.append(img_metrics)
            per_image_csv_rows.append({
                "variant": var_name,
                **{k: v for k, v in img_metrics.items() if k != "gt_match"},
            })
        agg = aggregate_metrics(per_image)
        size_recall = size_group_recall(all_names, variant_dets[var_name])
        agg["size_group_recall"] = size_recall
        py_metrics[var_name] = agg

    # Save per-image CSV
    csv_path = stage2_dir / "per_image.csv"
    if per_image_csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_image_csv_rows)

    # Bootstrap confidence intervals (best variant vs baseline)
    # Determine best non-baseline variant by FMeasure
    best_var = max(
        [v for v in passed_variants if v != "baseline"],
        key=lambda v: py_metrics[v]["fmeasure"]
    )
    print(f"\nBest variant: {best_var}")
    bootstrap_result = bootstrap_compare(
        all_names, baseline_dets, variant_dets[best_var], samples=10_000)

    # Small ellipse detailed analysis
    small_analysis = _analyze_small_ellipses(all_names, baseline_dets,
                                             variant_dets, variant_dets[best_var])

    # Oracle recall (union of all scale candidates)
    oracle_recall = _compute_oracle_recall(all_names, baseline_dets, branch_dets)

    # Generate case visualizations
    cases = generate_case_visualizations(
        all_names, baseline_dets, variant_dets[best_var], stage2_dir)

    # New candidate precision per branch
    branch_precision = {}
    for branch_name, _, _ in SCALE_BRANCHES:
        total_new = 0
        total_matched = 0
        for name in all_names:
            bl_dets = baseline_dets.get(name, [])
            br_dets = branch_dets[branch_name].get(name, [])
            new = [d for d in br_dets if not match_any(d, bl_dets)]
            gt = read_gt(GT_DIR / f"gt_{name}.txt")
            total_new += len(new)
            total_matched += sum(match_any(d, gt) for d in new)
        branch_precision[branch_name] = total_matched / total_new if total_new else 0.0

    # Verify baseline reproducibility
    baseline_check = _verify_baseline(cpp_metrics.get("baseline", {}))

    # Assemble summary
    summary = {
        "stage": 2,
        "best_variant": best_var,
        "cpp_metrics": cpp_metrics,
        "python_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "gt_match"}
                           for k, v in py_metrics.items()},
        "bootstrap": bootstrap_result,
        "oracle_recall": oracle_recall,
        "small_analysis": small_analysis,
        "branch_new_candidate_precision": branch_precision,
        "baseline_reproducibility": baseline_check,
        "cases": cases,
    }
    (stage2_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Generate report
    _generate_stage2_report(summary, stage2_dir)

    # Print final verdict
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    bl_py = py_metrics["baseline"]
    bv_py = py_metrics[best_var]
    print(f"Baseline:       P={bl_py['precision']:.6f} R={bl_py['recall']:.6f} "
          f"F={bl_py['fmeasure']:.6f}")
    print(f"Best ({best_var}): P={bv_py['precision']:.6f} R={bv_py['recall']:.6f} "
          f"F={bv_py['fmeasure']:.6f}")
    print(f"FMeasure delta: {bv_py['fmeasure'] - bl_py['fmeasure']:+.6f}")
    print(f"Recall delta:   {bv_py['recall'] - bl_py['recall']:+.6f}")
    print(f"Bootstrap F1 95% CI: {bootstrap_result['f1_delta_ci95']}")
    print(f"Bootstrap Recall 95% CI: {bootstrap_result['recall_delta_ci95']}")

    # Verdict
    ci_lower = bootstrap_result['f1_delta_ci95'][0]
    if ci_lower > 0:
        verdict = "已证明有效 (Proven Effective)"
    else:
        verdict = "有潜力 (Promising, not yet proven)"
    print(f"\nVerdict: {verdict}")

    return summary


def _analyze_small_ellipses(names: list[str],
                            baseline_dets: dict[str, list[Ellipse]],
                            all_variants: dict[str, dict[str, list[Ellipse]]],
                            best_dets: dict[str, list[Ellipse]]) -> dict:
    """Detailed analysis of small ellipse detection changes."""
    newly_recovered = 0
    displaced = 0
    still_missed = 0
    still_matched = 0

    for name in names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        small_gts = [e for e in gt if e.is_small]
        bl_list = baseline_dets.get(name, [])
        bv_list = best_dets.get(name, [])
        for sg in small_gts:
            in_bl = match_any(sg, bl_list)
            in_bv = match_any(sg, bv_list)
            if not in_bl and in_bv:
                newly_recovered += 1
            elif in_bl and not in_bv:
                displaced += 1
            elif not in_bl and not in_bv:
                still_missed += 1
            else:
                still_matched += 1

    total_small = newly_recovered + displaced + still_missed + still_matched
    return {
        "total_small_gt": total_small,
        "newly_recovered": newly_recovered,
        "displaced": displaced,
        "still_missed": still_missed,
        "still_matched": still_matched,
        "baseline_small_recall": (still_matched + displaced) / total_small if total_small else 0,
        "variant_small_recall": (still_matched + newly_recovered) / total_small if total_small else 0,
    }


def _compute_oracle_recall(names: list[str],
                           baseline_dets: dict[str, list[Ellipse]],
                           branch_dets: dict[str, dict[str, list[Ellipse]]]) -> dict:
    """Compute oracle recall: union of all scale branch candidates."""
    # Build union per image
    union_dets = {}
    for name in names:
        all_candidates = list(baseline_dets.get(name, []))
        for bn, _, _ in SCALE_BRANCHES:
            all_candidates.extend(branch_dets[bn].get(name, []))
        # Dedup (keep first occurrence)
        deduped = []
        for c in all_candidates:
            if not match_any(c, deduped):
                deduped.append(c)
        union_dets[name] = deduped

    total_gt = 0
    total_matched = 0
    for name in names:
        gt = read_gt(GT_DIR / f"gt_{name}.txt")
        total_gt += len(gt)
        total_matched += sum(match_any(t, union_dets.get(name, [])) for t in gt)

    return {
        "oracle_recall": total_matched / total_gt if total_gt else 0,
        "total_gt": total_gt,
        "total_matched": total_matched,
    }


def _verify_baseline(cpp_metrics: dict) -> dict:
    """Check if baseline metrics reproduce expected values."""
    if not cpp_metrics:
        return {"reproduced": False, "error": "no metrics"}
    p_ok = abs(cpp_metrics.get("Precision", 0) - BASELINE_PRECISION) < 0.001
    r_ok = abs(cpp_metrics.get("Recall", 0) - BASELINE_RECALL) < 0.001
    f_ok = abs(cpp_metrics.get("FMeasure", 0) - BASELINE_FMEASURE) < 0.001
    return {
        "reproduced": p_ok and r_ok and f_ok,
        "expected": {"Precision": BASELINE_PRECISION, "Recall": BASELINE_RECALL,
                     "FMeasure": BASELINE_FMEASURE},
        "actual": {k: cpp_metrics.get(k) for k in ["Precision", "Recall", "FMeasure"]},
    }


def _generate_stage2_report(summary: dict, output_dir: Path) -> None:
    """Generate comprehensive Chinese Markdown report."""
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    cpp = summary["cpp_metrics"]
    py_m = summary["python_metrics"]
    best = summary["best_variant"]
    bs = summary["bootstrap"]
    oracle = summary["oracle_recall"]
    small = summary["small_analysis"]
    branch_prec = summary["branch_new_candidate_precision"]
    bl_repro = summary.get("baseline_reproducibility", {})

    lines = [
        "# 小椭圆 FPN 式多尺度验证实验报告",
        "",
        f"**最终判定：** {summary.get('verdict', '见下文')}",
        "",
        "## 实验设计",
        "",
        "使用图像金字塔（INTER_CUBIC 2×/3× 放大）验证：将小椭圆放大后，",
        "现有 AAMED 是否能生成原尺度无法检测的有效候选，",
        "并通过跨尺度一致性融合同时提高全局 Recall 与 FMeasure。",
        "",
        "### 尺度分支",
        "",
        "| 分支 | Scale | T_val |",
        "| --- | ---: | ---: |",
    ]
    for bn, s, t in SCALE_BRANCHES:
        lines.append(f"| `{bn}` | {s}× | {t} |")

    lines.extend([
        "",
        "### 融合变体",
        "",
        "1. `baseline` — 原始 1× T=0.77",
        "2. `baseline + scale2_t77 union` — 直接合并 2× T=0.77 候选",
        "3. `baseline + all-scale union` — 合并所有 4 个尺度分支候选",
        "4. `baseline + scale2 stable` — 2× 双阈值一致性候选",
        "5. `baseline + cross-scale` — 跨 2×/3× 尺度一致性候选",
        "6. `baseline + cross-scale strict` — 3+ 分支支持 + 跨尺度",
        "",
        "## 全局结果 (C++ aamed_eval)",
        "",
        "| 变体 | Precision | Recall | FMeasure | 检测数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    var_order = ["baseline", "union_scale2_t77", "union_all_scale",
                 "stable_scale2", "cross_scale", "cross_scale_strict",
                 "adaptive_iou", "cross_scale_adaptive"]
    for vn in var_order:
        if vn in cpp:
            m = cpp[vn]
            lines.append(
                f"| {vn} | {m['Precision']:.6f} | {m['Recall']:.6f} | "
                f"{m['FMeasure']:.6f} | {int(m.get('DetectedCount', 0))} |"
            )

    lines.extend([
        "",
        "## Python 原生评估（含分组 Recall）",
        "",
        "| 变体 | Precision | Recall | FMeasure | <15px | 15-30 | 30-60 | 60+ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for vn in var_order:
        if vn in py_m:
            m = py_m[vn]
            sr = m.get("size_group_recall", {})
            lines.append(
                f"| {vn} | {m['precision']:.6f} | {m['recall']:.6f} | "
                f"{m['fmeasure']:.6f} | {sr.get('radius_lt15', 0):.4f} | "
                f"{sr.get('radius_15_30', 0):.4f} | {sr.get('radius_30_60', 0):.4f} | "
                f"{sr.get('radius_60_plus', 0):.4f} |"
            )

    lines.extend([
        "",
        "## 小椭圆分析",
        "",
        f"- 小椭圆 GT 总数：{small['total_small_gt']}",
        f"- Baseline 已匹配：{small['still_matched'] + small['displaced']}",
        f"- 新恢复：{small['newly_recovered']}",
        f"- 被挤掉：{small['displaced']}",
        f"- 仍未匹配：{small['still_missed']}",
        f"- Baseline 小椭圆 Recall：{small['baseline_small_recall']:.6f}",
        f"- 最优变体小椭圆 Recall：{small['variant_small_recall']:.6f}",
        "",
        "## Oracle Recall",
        "",
        f"- 全部尺度候选 union Recall：{oracle['oracle_recall']:.6f}",
        f"- 已匹配：{oracle['total_matched']}/{oracle['total_gt']}",
        "",
        "## 各分支新增候选 Precision",
        "",
        "| 分支 | 新增候选 Precision |",
        "| --- | ---: |",
    ])
    for bn, _, _ in SCALE_BRANCHES:
        lines.append(f"| `{bn}` | {branch_prec.get(bn, 0):.6f} |")

    lines.extend([
        "",
        "## Bootstrap 95% 置信区间（图像级重采样 10,000 次）",
        "",
        f"- FMeasure 差值 95% CI：`{bs['f1_delta_ci95']}`",
        f"- FMeasure 差值均值：`{bs['f1_delta_mean']:.6f}`",
        f"- Recall 差值 95% CI：`{bs['recall_delta_ci95']}`",
        f"- Recall 差值均值：`{bs['recall_delta_mean']:.6f}`",
        "",
        "## Baseline 复现检查",
        "",
        f"- 复现成功：{bl_repro.get('reproduced', 'N/A')}",
        f"- 期望：P={BASELINE_PRECISION:.6f} R={BASELINE_RECALL:.6f} "
        f"F={BASELINE_FMEASURE:.6f}",
        f"- 实际：{bl_repro.get('actual', 'N/A')}",
        "",
        "## 判定",
    ])

    ci_lower = bs['f1_delta_ci95'][0]
    bl_py = py_m.get("baseline", {})
    bv_py = py_m.get(best, {})
    if ci_lower > 0 and bv_py.get("recall", 0) > BASELINE_RECALL \
       and bv_py.get("fmeasure", 0) > BASELINE_FMEASURE:
        lines.append("Bootstrap 95% CI 下界 > 0 且 Recall/FMeasure 均超过 baseline → **已证明有效**。")
    else:
        lines.append("Bootstrap 95% CI 包含 0 或指标未全面超越 → **有潜力，但尚未严格证明**。")

    lines.extend([
        "",
        "## 案例可视化",
        "",
        "绿色 = GT，红色 = baseline，蓝色 = 最优变体。",
        "",
    ])
    for case in summary.get("cases", []):
        lines.append(f"- `{case}`")

    lines.extend([
        "",
        "## 复现方法",
        "",
        "```powershell",
        r".\build.bat",
        r"python tools\run_scc_small_fpn_eval.py --stage 2 --skip-build",
        "```",
    ])

    report_path = DOC_DIR / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def run_correctness_tests() -> bool:
    """Run correctness tests for remapping and pipeline invariants."""
    print("\n--- Correctness Tests ---")
    all_ok = True

    # Test 1: 1x remap identity
    print("Test 1: 1x remap identity...")
    test_ellipse = Ellipse(cx=100.0, cy=200.0, a=50.0, b=30.0, theta=0.5)
    remapped = remap_detections([test_ellipse], 1)
    err = max(
        abs(remapped[0].cx - test_ellipse.cx),
        abs(remapped[0].cy - test_ellipse.cy),
        abs(remapped[0].a - test_ellipse.a),
        abs(remapped[0].b - test_ellipse.b),
        abs(remapped[0].theta - test_ellipse.theta),
    )
    ok = err < 1e-10
    print(f"  1x remap error: {err:.2e} {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 2: Round-trip (2x up, remap down, 2x up again)
    print("Test 2: Round-trip 2x/3x...")
    for scale in [2, 3]:
        # Simulate: original ellipse -> upscale -> detect -> remap -> verify
        # We test that remap(detect_at_scale(upscale(original))) preserves
        # the geometry of small ellipses correctly
        orig = Ellipse(cx=50.0, cy=60.0, a=8.0, b=6.0, theta=0.3)
        # upscale: center and axes multiply by scale
        upscaled = Ellipse(
            cx=(orig.cx - 1.0) * scale + 1.0,
            cy=(orig.cy - 1.0) * scale + 1.0,
            a=orig.a * scale,
            b=orig.b * scale,
            theta=orig.theta,
        )
        # remap back
        back = remap_detections([upscaled], scale)[0]
        err = max(
            abs(back.cx - orig.cx),
            abs(back.cy - orig.cy),
            abs(back.a - orig.a),
            abs(back.b - orig.b),
            abs(back.theta - orig.theta),
        )
        ok = err < 1e-10
        print(f"  {scale}x round-trip error: {err:.2e} {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # Test 3: fled.txt write/read round-trip
    print("Test 3: fled.txt write/read round-trip...")
    test_ellipses = [
        Ellipse(cx=100.5, cy=200.3, a=45.7, b=32.1, theta=0.8),
        Ellipse(cx=150.2, cy=180.9, a=25.0, b=25.0, theta=1.57),
    ]
    tmp_path = OUTPUT / "test_roundtrip.fled.txt"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    write_fled(tmp_path, test_ellipses, 12.345)
    _, read_back = read_fled(tmp_path)
    ok = len(read_back) == len(test_ellipses)
    for a, b in zip(test_ellipses, read_back):
        err = max(
            abs(a.cx - b.cx), abs(a.cy - b.cy),
            abs(a.a - b.a), abs(a.b - b.b),
            abs(a.theta - b.theta),
        )
        if err > 1e-5:
            ok = False
            print(f"    round-trip error: {err:.2e}")
    print(f"  fled.txt round-trip: {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok
    tmp_path.unlink(missing_ok=True)

    # Test 4: overlap identity (numerical integration, tolerance 0.01)
    print("Test 4: overlap identity...")
    e = Ellipse(cx=50, cy=50, a=20, b=15, theta=0)
    iou = overlap(e, e)
    ok = abs(iou - 1.0) < 0.01
    print(f"  self-IoU = {iou:.6f} {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 5: filter_small_candidates
    print("Test 5: small candidate filter...")
    mixed = [
        Ellipse(cx=10, cy=10, a=5, b=5, theta=0),     # r=5  < 20: keep
        Ellipse(cx=20, cy=20, a=30, b=30, theta=0),    # r=30 >= 20: discard
        Ellipse(cx=30, cy=30, a=19, b=19, theta=0),    # r=19 < 20: keep
        Ellipse(cx=40, cy=40, a=1, b=1, theta=0),      # r=1  < 20: keep
    ]
    filtered = filter_small_candidates(mixed, 20.0)
    ok = len(filtered) == 3
    print(f"  kept {len(filtered)}/4 {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 6: empty detection handling
    print("Test 6: empty detection handling...")
    empty_dets = filter_small_candidates([], 20.0)
    ok = len(empty_dets) == 0
    print(f"  empty input: {len(empty_dets)} dets {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 7: IoU graph construction
    print("Test 7: IoU graph / clustering...")
    # Two close circles (d≈0.7px, IoU>0.8) + one far circle
    candidates = [
        ("a", Ellipse(cx=50.0, cy=50.0, a=10, b=10, theta=0)),
        ("b", Ellipse(cx=50.5, cy=50.5, a=10, b=10, theta=0)),   # overlaps a (IoU~0.95)
        ("c", Ellipse(cx=200, cy=200, a=10, b=10, theta=0)),      # far from a,b
    ]
    graph = build_iou_graph(candidates)
    clusters = find_clusters(graph, len(candidates))
    ok = len(clusters) == 2  # {a,b} and {c}
    print(f"  clusters: {len(clusters)} (expected 2) {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 8: fusion dedup (no duplicate addition)
    print("Test 8: fusion dedup...")
    baseline = [Ellipse(cx=50, cy=50, a=10, b=10, theta=0)]
    candidates = [Ellipse(cx=50.5, cy=50.5, a=10, b=10, theta=0)]  # very close
    result = fusion_union(baseline, {"test": candidates})
    ok = len(result) == 1  # candidate should be deduped
    print(f"  dedup result: {len(result)} (expected 1) {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Always run correctness tests first
    if not run_correctness_tests():
        print("Correctness tests failed. Aborting.")
        return 1

    # Build if needed
    ensure_build(args.skip_build)

    if args.stage == 1:
        summary = run_stage1(args)
        print(f"\nStage 1 summary: {OUTPUT / 'stage1' / 'stage1_summary.json'}")
    else:
        # Try to load stage 1 summary
        stage1_path = OUTPUT / "stage1" / "stage1_summary.json"
        stage1_summary = None
        if stage1_path.exists():
            stage1_summary = json.loads(stage1_path.read_text(encoding="utf-8"))
        summary = run_stage2(args, stage1_summary)
        print(f"\nStage 2 summary: {OUTPUT / 'stage2' / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
