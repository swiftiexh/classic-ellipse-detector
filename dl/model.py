from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class PredictionHead(nn.Sequential):
    def __init__(self, channels: int, outputs: int, bias: float | None = None):
        layers = [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, outputs, 1)]
        super().__init__(*layers)
        if bias is not None:
            nn.init.constant_(self[-1].bias, bias)


class EllipseCenterNet(nn.Module):
    """Single-scale lightweight baseline with explicit extension points."""

    def __init__(self, slots: int = 5, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.neck = nn.Sequential(
            nn.Conv2d(576, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self._up(128, 128),
            self._up(128, 96),
            self._up(96, 64),
        )
        self.quality_refiner = nn.Identity()
        self.heads = nn.ModuleDict(
            {
                "heatmap": PredictionHead(64, slots, bias=-2.19),
                "offset": PredictionHead(64, slots * 2),
                "axes": PredictionHead(64, slots * 2),
                "angle": PredictionHead(64, slots * 2),
            }
        )
        self.slots = slots

    @staticmethod
    def _up(source: int, target: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(source, target, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.quality_refiner(self.neck(self.backbone(image)))
        outputs = {name: head(features) for name, head in self.heads.items()}
        for name in ("offset", "axes", "angle"):
            batch, _, height, width = outputs[name].shape
            outputs[name] = outputs[name].view(batch, self.slots, 2, height, width)
        return outputs


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
