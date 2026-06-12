from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Keep logarithms in float32: 1 - 1e-4 rounds to 1 in float16.
    prediction = logits.float().sigmoid().clamp(1e-4, 1.0 - 1e-4)
    target = target.float()
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -(prediction.log()) * (1.0 - prediction).pow(2) * positive
    negative_loss = -((1.0 - prediction).log()) * prediction.pow(2) * negative_weight * negative
    count = positive.sum().clamp(min=1.0)
    return (positive_loss.sum() + negative_loss.sum()) / count


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    return (F.smooth_l1_loss(prediction, target, reduction="none") * expanded).sum() / expanded.sum().clamp(min=1.0)


def ellipse_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    heatmap = focal_heatmap_loss(outputs["heatmap"], batch["heatmap"])
    offset = masked_smooth_l1(outputs["offset"], batch["offset"], batch["mask"])
    axes = masked_smooth_l1(outputs["axes"], batch["axes"], batch["mask"])
    normalized_angle = F.normalize(outputs["angle"], dim=2, eps=1e-6)
    angle = masked_smooth_l1(normalized_angle, batch["angle"], batch["mask"])
    total = weights["heatmap"] * heatmap + weights["offset"] * offset + weights["axes"] * axes + weights["angle"] * angle
    return {"total": total, "heatmap": heatmap, "offset": offset, "axes": axes, "angle": angle}
