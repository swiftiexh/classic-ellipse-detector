from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import project_root
from .geometry import EllipseRecord, LetterboxTransform


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    folder: str
    gt_prefix: str
    convention: str
    iou_threshold: float


SPECS = {
    "prasad": DatasetSpec("prasad", "Prasad Images - Dataset Prasad", "gt_", "xy_rad", 0.8),
    "random": DatasetSpec("random", "Random Images - Dataset #1", "gt_", "xy_deg", 0.8),
    "smartphone": DatasetSpec("smartphone", "Smartphone Images - Dataset #2", "gt_", "xy_deg", 0.8),
    "concentric": DatasetSpec("concentric", "Concentric Ellipses - Dataset Synthetic", "", "concentric_deg", 0.95),
    "concurrent": DatasetSpec("concurrent", "Concurrent Ellipses - Dataset Synthetic", "", "concentric_deg", 0.95),
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    name: str
    image_path: Path
    gt_path: Path


def list_samples(dataset: str) -> list[Sample]:
    spec = SPECS[dataset]
    root = project_root() / "dataset" / spec.folder
    names_path = root / "imagenames.txt"
    if names_path.exists():
        names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = sorted(path.name for path in (root / "images").iterdir() if path.is_file())
    return [Sample(dataset, name, _resolve_image(root / "images", name), _resolve_gt(root / "gt", spec, name)) for name in names]


def read_ground_truth(sample: Sample) -> list[EllipseRecord]:
    lines = sample.gt_path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    records = [_convert([float(value) for value in line.split()], SPECS[sample.dataset].convention) for line in lines[1 : count + 1]]
    if len(records) != count:
        raise ValueError(f"Expected {count} ellipses in {sample.gt_path}, got {len(records)}")
    return records


def _convert(raw: list[float], convention: str) -> EllipseRecord:
    if convention == "xy_rad":
        ellipse = EllipseRecord(raw[0], raw[1], raw[2], raw[3], raw[4])
    elif convention == "xy_deg":
        ellipse = EllipseRecord(raw[0], raw[1], raw[2], raw[3], math.radians(raw[4]))
    elif convention == "concentric_deg":
        ellipse = EllipseRecord(raw[1], raw[0], raw[3], raw[2], -math.radians(raw[4]))
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return ellipse.canonical()


def _resolve_image(images_dir: Path, name: str) -> Path:
    direct = images_dir / name
    if direct.exists():
        return direct
    for extension in (".jpg", ".png", ".bmp", ".jpeg", ".tif", ".tiff"):
        candidate = images_dir / f"{name}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(direct)


def _resolve_gt(gt_dir: Path, spec: DatasetSpec, name: str) -> Path:
    for base in (name, Path(name).stem):
        candidate = gt_dir / f"{spec.gt_prefix}{base}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(gt_dir / f"{spec.gt_prefix}{name}.txt")


def select_overfit_samples(datasets: Iterable[str], per_dataset: int, seed: int) -> list[Sample]:
    selected: list[Sample] = []
    for dataset in datasets:
        samples = [(sample, len(read_ground_truth(sample))) for sample in list_samples(dataset)]
        nonempty = [(sample, count) for sample, count in samples if count > 0]
        nonempty.sort(key=lambda item: (item[1], item[0].name))
        if len(nonempty) < per_dataset:
            raise ValueError(f"{dataset} only has {len(nonempty)} non-empty samples")
        rng = random.Random(f"{seed}:{dataset}")
        bins = np.array_split(np.arange(len(nonempty)), per_dataset)
        selected.extend(nonempty[rng.choice(list(indices))][0] for indices in bins)
    return selected


def split_samples(dataset: str, validation_fraction: float, test_fraction: float, seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    samples = list_samples(dataset)
    rng = random.Random(f"{seed}:{dataset}:split")
    rng.shuffle(samples)
    test_count = round(len(samples) * test_fraction)
    validation_count = round(len(samples) * validation_fraction)
    test = samples[:test_count]
    validation = samples[test_count : test_count + validation_count]
    train = samples[test_count + validation_count :]
    if not train or not validation or not test:
        raise ValueError(f"Split for {dataset} produced an empty partition")
    return train, validation, test


class EllipseDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int = 512, stride: int = 4, slots: int = 5):
        self.samples = samples
        self.input_size = input_size
        self.stride = stride
        self.slots = slots

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.image_path)
        transform = LetterboxTransform.create(image.shape[1], image.shape[0], self.input_size)
        ellipses = [transform.ellipse(ellipse).canonical() for ellipse in read_ground_truth(sample)]
        targets = encode_targets(ellipses, self.input_size, self.stride, self.slots)
        rgb = cv2.cvtColor(transform.image(image), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        return {
            "image": tensor,
            **{key: torch.from_numpy(value) for key, value in targets.items()},
            "dataset": sample.dataset,
            "name": sample.name,
            "original_size": torch.tensor([image.shape[1], image.shape[0]], dtype=torch.int32),
        }


def encode_targets(ellipses: list[EllipseRecord], input_size: int, stride: int, slots: int) -> dict[str, np.ndarray]:
    out_size = input_size // stride
    heatmap = np.zeros((slots, out_size, out_size), dtype=np.float32)
    offset = np.zeros((slots, 2, out_size, out_size), dtype=np.float32)
    axes = np.zeros_like(offset)
    angle = np.zeros_like(offset)
    mask = np.zeros((slots, 1, out_size, out_size), dtype=np.float32)
    grouped: dict[tuple[int, int], list[EllipseRecord]] = defaultdict(list)
    for ellipse in ellipses:
        x, y = ellipse.cx / stride, ellipse.cy / stride
        cell = (int(x), int(y))
        if 0 <= cell[0] < out_size and 0 <= cell[1] < out_size:
            grouped[cell].append(ellipse.canonical())
    for (x, y), cell_ellipses in grouped.items():
        cell_ellipses.sort(key=lambda e: e.a * e.b, reverse=True)
        if len(cell_ellipses) > slots:
            raise ValueError(f"Cell {(x, y)} contains {len(cell_ellipses)} ellipses, but only {slots} slots are available")
        for slot, ellipse in enumerate(cell_ellipses):
            radius = max(1, int(_gaussian_radius(2.0 * ellipse.b / stride, 2.0 * ellipse.a / stride)))
            _draw_gaussian(heatmap[slot], (x, y), radius)
            offset[slot, :, y, x] = [ellipse.cx / stride - x, ellipse.cy / stride - y]
            axes[slot, :, y, x] = [math.log(max(ellipse.a / stride, 1e-4)), math.log(max(ellipse.b / stride, 1e-4))]
            angle[slot, :, y, x] = [math.cos(2.0 * ellipse.theta), math.sin(2.0 * ellipse.theta)]
            mask[slot, 0, y, x] = 1.0
    return {"heatmap": heatmap, "offset": offset, "axes": axes, "angle": angle, "mask": mask}


def _gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    a1, b1, c1 = 1.0, height + width, width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / 2.0
    a2, b2, c2 = 4.0, 2.0 * (height + width), (1.0 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))) / 2.0
    a3, b3, c3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width), (min_overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / (2.0 * a3)
    return min(r1, r2, r3)


def _draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    x = np.arange(diameter, dtype=np.float32) - radius
    gaussian = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2.0 * (diameter / 6.0) ** 2))
    cx, cy = center
    left, right = min(cx, radius), min(heatmap.shape[1] - cx - 1, radius)
    top, bottom = min(cy, radius), min(heatmap.shape[0] - cy - 1, radius)
    patch = heatmap[cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    np.maximum(patch, gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1], out=patch)
