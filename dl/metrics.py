from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import EllipseRecord, ellipse_overlap


def legacy_metrics(
    ground_truth: dict[str, list[EllipseRecord]],
    predictions: dict[str, list[EllipseRecord]],
    iou_threshold: float,
    score_threshold: float = 0.0,
) -> dict[str, float]:
    positive = detected = total_gt = 0
    for name, gt in ground_truth.items():
        det = [ellipse for ellipse in predictions.get(name, []) if ellipse.score >= score_threshold]
        gt_match = [0] * len(gt)
        det_match = [0] * len(det)
        for det_index, detection in enumerate(det):
            for gt_index, target in enumerate(gt):
                if ellipse_overlap(detection, target) > iou_threshold:
                    det_match[det_index] += 1
                    gt_match[gt_index] += 1
        num_true = sum(value > 0 for value in gt_match)
        num_det_true = sum(value > 0 for value in det_match)
        num_false = sum(value == 0 for value in det_match)
        positive += num_true
        detected += num_true + num_false + max(num_det_true - num_true, 0)
        total_gt += len(gt)
    precision = positive / detected if detected else 0.0
    recall = positive / total_gt if total_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "positive_matches": positive,
        "detected_count": detected,
        "ground_truth_count": total_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def standard_curve(
    ground_truth: dict[str, list[EllipseRecord]],
    predictions: dict[str, list[EllipseRecord]],
    iou_threshold: float,
) -> dict[str, object]:
    ranked = sorted(
        ((ellipse.score, name, ellipse) for name, values in predictions.items() for ellipse in values),
        key=lambda item: item[0],
        reverse=True,
    )
    matched = {name: [False] * len(values) for name, values in ground_truth.items()}
    tp: list[float] = []
    fp: list[float] = []
    thresholds: list[float] = []
    for score, name, detection in ranked:
        targets = ground_truth.get(name, [])
        best_index, best_iou = -1, iou_threshold
        for index, target in enumerate(targets):
            if not matched[name][index]:
                overlap = ellipse_overlap(detection, target)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
        is_true = best_index >= 0
        if is_true:
            matched[name][best_index] = True
        tp.append(float(is_true))
        fp.append(float(not is_true))
        thresholds.append(float(score))
    total_gt = sum(len(values) for values in ground_truth.values())
    cumulative_tp = np.cumsum(tp, dtype=np.float64)
    cumulative_fp = np.cumsum(fp, dtype=np.float64)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall = cumulative_tp / max(total_gt, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1)) if len(f1) else -1
    ap = interpolated_ap(recall, precision)
    return {
        "ap": ap,
        "best_f1": float(f1[best_index]) if best_index >= 0 else 0.0,
        "best_precision": float(precision[best_index]) if best_index >= 0 else 0.0,
        "best_recall": float(recall[best_index]) if best_index >= 0 else 0.0,
        "best_threshold": thresholds[best_index] if best_index >= 0 else 1.0,
        "thresholds": thresholds,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if not len(recall):
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def read_fled_predictions(path: str | Path) -> tuple[float, list[EllipseRecord]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0.0, []
    time_ms = float(lines[0])
    ellipses = []
    for line in lines[1:]:
        values = [float(value) for value in line.split()]
        if len(values) < 6 or int(values[0]) == 2:
            continue
        ellipses.append(
            EllipseRecord(
                cx=values[2] + 1.0,
                cy=values[1] + 1.0,
                a=values[3] / 2.0,
                b=values[4] / 2.0,
                theta=-np.deg2rad(values[5]),
            ).canonical()
        )
    return time_ms, ellipses


def save_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def predictions_from_fled(result_dir: str | Path, names: Iterable[str]) -> tuple[dict[str, list[EllipseRecord]], float]:
    result_dir = Path(result_dir)
    predictions: dict[str, list[EllipseRecord]] = {}
    total_ms = 0.0
    for name in names:
        time_ms, ellipses = read_fled_predictions(result_dir / f"{name}.fled.txt")
        predictions[name] = ellipses
        total_ms += time_ms
    return predictions, total_ms
