from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .data import EllipseDataset, Sample, read_ground_truth
from .geometry import EllipseRecord, LetterboxTransform, draw_ellipses
from .metrics import legacy_metrics, standard_curve


@torch.inference_mode()
def decode_outputs(
    outputs: dict[str, torch.Tensor],
    transforms: list[LetterboxTransform],
    stride: int,
    topk: int,
    score_threshold: float,
) -> list[list[EllipseRecord]]:
    heatmap = outputs["heatmap"].sigmoid()
    peaks = heatmap.eq(F.max_pool2d(heatmap, 3, stride=1, padding=1)) * heatmap
    batch, slots, height, width = peaks.shape
    k = min(topk, slots * height * width)
    scores, indices = torch.topk(peaks.reshape(batch, -1), k)
    decoded: list[list[EllipseRecord]] = []
    for batch_index in range(batch):
        ellipses = []
        for score, flat_index in zip(scores[batch_index].tolist(), indices[batch_index].tolist()):
            if score < score_threshold:
                continue
            slot = flat_index // (height * width)
            location = flat_index % (height * width)
            y, x = divmod(location, width)
            offset = outputs["offset"][batch_index, slot, :, y, x]
            axes = outputs["axes"][batch_index, slot, :, y, x].clamp(-6.0, 8.0).exp()
            angle = outputs["angle"][batch_index, slot, :, y, x]
            theta = 0.5 * math.atan2(float(angle[1]), float(angle[0]))
            letterboxed = EllipseRecord(
                (x + float(offset[0])) * stride,
                (y + float(offset[1])) * stride,
                float(axes[0]) * stride,
                float(axes[1]) * stride,
                theta,
                float(score),
            ).canonical()
            ellipses.append(transforms[batch_index].inverse_ellipse(letterboxed))
        decoded.append(ellipses)
    return decoded


@torch.inference_mode()
def predict_samples(
    model: torch.nn.Module,
    samples: list[Sample],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, list[EllipseRecord]]:
    model.eval()
    dataset = EllipseDataset(samples, config["input_size"], config["output_stride"], config["slots"])
    predictions: dict[str, list[EllipseRecord]] = {}
    for sample, item in zip(samples, dataset):
        width, height = item["original_size"].tolist()
        transform = LetterboxTransform.create(width, height, config["input_size"])
        outputs = model(item["image"].unsqueeze(0).to(device))
        predictions[sample.name] = decode_outputs(
            outputs, [transform], config["output_stride"], config["topk"], config["score_threshold"]
        )[0]
    return predictions


def evaluate_predictions(
    samples: list[Sample],
    predictions: dict[str, list[EllipseRecord]],
    iou_threshold: float,
    selected_threshold: float | None = None,
) -> dict[str, Any]:
    ground_truth = {sample.name: read_ground_truth(sample) for sample in samples}
    standard = standard_curve(ground_truth, predictions, iou_threshold)
    threshold = standard["best_threshold"] if selected_threshold is None else selected_threshold
    legacy = legacy_metrics(ground_truth, predictions, iou_threshold, threshold)
    return {"selected_threshold": threshold, "legacy": legacy, "standard": standard}


def save_predictions(path: str | Path, predictions: dict[str, list[EllipseRecord]]) -> None:
    payload = {
        name: [
            {"cx": e.cx, "cy": e.cy, "a": e.a, "b": e.b, "theta": e.theta, "score": e.score}
            for e in ellipses
        ]
        for name, ellipses in predictions.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_predictions(path: str | Path) -> dict[str, list[EllipseRecord]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [EllipseRecord(**record).canonical() for record in records] for name, records in payload.items()}


def save_curves(path: str | Path, standard: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(standard["recall"], standard["precision"])
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"PR, AP={standard['ap']:.4f}", xlim=(0, 1), ylim=(0, 1))
    axes[1].plot(standard["thresholds"], standard["f1"])
    axes[1].set(xlabel="Score threshold", ylabel="F1", title=f"Best F1={standard['best_f1']:.4f}", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_visualizations(
    directory: str | Path,
    samples: list[Sample],
    predictions: dict[str, list[EllipseRecord]],
    limit: int,
    threshold: float,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for sample in samples[:limit]:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        canvas = draw_ellipses(image, read_ground_truth(sample), (0, 255, 0))
        canvas = draw_ellipses(canvas, [e for e in predictions.get(sample.name, []) if e.score >= threshold], (0, 0, 255))
        cv2.imwrite(str(directory / f"{sample.dataset}__{sample.name}.png"), canvas)
