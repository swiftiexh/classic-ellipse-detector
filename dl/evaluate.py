from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config, project_root
from .data import list_samples, read_ground_truth
from .metrics import legacy_metrics, predictions_from_fled, save_metrics
from .model import EllipseCenterNet
from .runtime import evaluate_predictions, predict_samples, save_curves, save_predictions, save_visualizations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl_baseline.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", choices=["prasad", "random", "smartphone", "concentric", "concurrent"])
    parser.add_argument("--fled-dir", help="Evaluate an existing traditional detector output directory")
    parser.add_argument("--checkpoint-split", choices=["train", "validation", "test"])
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    datasets = [args.dataset] if args.dataset else config["datasets"]
    output = Path(args.output or (project_root() / "output" / "dl" / "evaluation"))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if args.checkpoint_split:
        if checkpoint is None:
            raise ValueError("--checkpoint-split requires --checkpoint")
        lookup = {(sample.dataset, sample.name): sample for dataset in config["datasets"] for sample in list_samples(dataset)}
        samples = [lookup[(item["dataset"], item["name"])] for item in checkpoint["sample_splits"][args.checkpoint_split]]
    else:
        samples = [sample for dataset in datasets for sample in list_samples(dataset)]
    if args.fled_dir:
        if len(datasets) != 1:
            raise ValueError("--fled-dir requires exactly one --dataset")
        predictions, total_ms = predictions_from_fled(args.fled_dir, [sample.name for sample in samples])
        ground_truth = {sample.name: read_ground_truth(sample) for sample in samples}
        metrics = {"legacy": legacy_metrics(ground_truth, predictions, config["evaluate"]["iou_threshold"])}
        metrics["average_time_ms"] = total_ms / max(len(samples), 1)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --fled-dir is supplied")
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        checkpoint = checkpoint or torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = EllipseCenterNet(config["slots"], False).to(device)
        model.load_state_dict(checkpoint["model"])
        predictions = predict_samples(model, samples, config, device)
        selected_threshold = checkpoint.get("selected_threshold") if args.checkpoint_split == "test" else None
        metrics = evaluate_predictions(samples, predictions, config["evaluate"]["iou_threshold"], selected_threshold)
        save_predictions(output / "predictions.json", predictions)
        save_curves(output / "curves.png", metrics["standard"])
        save_visualizations(output / "visualizations", samples, predictions, config["evaluate"]["save_visualizations"], metrics["standard"]["best_threshold"])
    save_metrics(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
