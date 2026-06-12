from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from .config import project_root
from .data import EllipseDataset, encode_targets, list_samples, read_ground_truth, split_samples
from .geometry import EllipseRecord, LetterboxTransform, ellipse_overlap, horizontal_flip
from .losses import ellipse_losses, focal_heatmap_loss
from .metrics import legacy_metrics, predictions_from_fled
from .model import EllipseCenterNet, parameter_count
from .runtime import decode_outputs


class GeometryTests(unittest.TestCase):
    def test_canonical_angle_equivalence(self) -> None:
        source = EllipseRecord(20, 30, 5, 9, 0.3)
        canonical = source.canonical()
        self.assertGreaterEqual(canonical.a, canonical.b)
        self.assertGreaterEqual(canonical.theta, -math.pi / 2)
        self.assertLess(canonical.theta, math.pi / 2)
        self.assertGreater(ellipse_overlap(source, canonical), 0.999)

    def test_letterbox_round_trip(self) -> None:
        transform = LetterboxTransform.create(640, 480, 512)
        ellipse = EllipseRecord(321.2, 203.4, 81.0, 32.0, -0.7)
        restored = transform.inverse_ellipse(transform.ellipse(ellipse))
        for field in ("cx", "cy", "a", "b", "theta"):
            self.assertAlmostEqual(getattr(ellipse, field), getattr(restored, field), places=5)

    def test_horizontal_flip_round_trip(self) -> None:
        ellipse = EllipseRecord(25, 20, 12, 4, 0.6)
        restored = horizontal_flip(horizontal_flip(ellipse, 100), 100)
        self.assertGreater(ellipse_overlap(ellipse, restored), 0.999)


class DataTests(unittest.TestCase):
    def test_all_five_dataset_conventions_load(self) -> None:
        expected = {"prasad": 1, "random": 12, "smartphone": 2, "concentric": 2, "concurrent": 2}
        for dataset, count in expected.items():
            records = read_ground_truth(list_samples(dataset)[0])
            self.assertEqual(len(records), count)
            self.assertTrue(all(record.a >= record.b > 0 for record in records))

    def test_five_same_cell_targets_are_preserved(self) -> None:
        ellipses = [EllipseRecord(100.2, 101.7, 30 - index, 10 - index, 0.1 * index) for index in range(5)]
        targets = encode_targets(ellipses, 512, 4, 5)
        self.assertEqual(int(targets["mask"].sum()), 5)
        self.assertEqual(sum(float(targets["heatmap"][slot].max()) == 1.0 for slot in range(5)), 5)

    def test_formal_split_is_disjoint_and_complete(self) -> None:
        train, validation, test = split_samples("prasad", 0.15, 0.15, 3407)
        names = [{sample.name for sample in split} for split in (train, validation, test)]
        self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
        self.assertEqual(sum(len(split) for split in (train, validation, test)), len(list_samples("prasad")))

    def test_encode_decode_ground_truth_tensor(self) -> None:
        ellipse = EllipseRecord(100.2, 101.7, 30, 10, 0.4)
        targets = encode_targets([ellipse], 512, 4, 5)
        outputs = {
            "heatmap": torch.full((1, 5, 128, 128), -20.0),
            "offset": torch.from_numpy(targets["offset"]).unsqueeze(0),
            "axes": torch.from_numpy(targets["axes"]).unsqueeze(0),
            "angle": torch.from_numpy(targets["angle"]).unsqueeze(0),
        }
        y, x = int(ellipse.cy / 4), int(ellipse.cx / 4)
        outputs["heatmap"][0, 0, y, x] = 20.0
        decoded = decode_outputs(outputs, [LetterboxTransform.create(512, 512, 512)], 4, 1, 0.1)[0][0]
        self.assertGreater(ellipse_overlap(ellipse, decoded), 0.999)


class ModelTests(unittest.TestCase):
    def test_extreme_amp_heatmap_loss_is_finite(self) -> None:
        logits = torch.tensor([[[[100.0, -100.0]]]], dtype=torch.float16)
        target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)
        self.assertTrue(torch.isfinite(focal_heatmap_loss(logits, target)))

    def test_forward_backward_and_parameter_budget(self) -> None:
        model = EllipseCenterNet(5, False)
        image = torch.randn(2, 3, 128, 128)
        outputs = model(image)
        self.assertEqual(tuple(outputs["heatmap"].shape), (2, 5, 32, 32))
        self.assertEqual(tuple(outputs["axes"].shape), (2, 5, 2, 32, 32))
        targets = encode_targets([EllipseRecord(50, 60, 20, 10, 0.2)], 128, 4, 5)
        batch = {key: torch.from_numpy(value).unsqueeze(0).repeat(2, *([1] * value.ndim)) for key, value in targets.items()}
        losses = ellipse_losses(outputs, batch, {"heatmap": 1.0, "offset": 1.0, "axes": 0.5, "angle": 0.2})
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertLess(parameter_count(model), 2_000_000)


class LegacyParityTests(unittest.TestCase):
    def test_smartphone_baseline_matches_cpp_report(self) -> None:
        result_dir = project_root() / "output" / "smartphone" / "baseline"
        report_path = result_dir / "eval_report.txt"
        if not report_path.exists():
            self.skipTest("Existing Smartphone baseline output is unavailable")
        samples = list_samples("smartphone")
        predictions, _ = predictions_from_fled(result_dir, [sample.name for sample in samples])
        ground_truth = {sample.name: read_ground_truth(sample) for sample in samples}
        actual = legacy_metrics(ground_truth, predictions, 0.8)
        report = {}
        for line in report_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            report[key] = float(value.strip())
        self.assertAlmostEqual(actual["precision"], report["Precision"], places=6)
        self.assertAlmostEqual(actual["recall"], report["Recall"], places=6)
        self.assertAlmostEqual(actual["f1"], report["FMeasure"], places=6)


if __name__ == "__main__":
    unittest.main()
