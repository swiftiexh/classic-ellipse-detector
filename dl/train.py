from __future__ import annotations

import argparse
import csv
import json
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_config, run_dir
from .data import EllipseDataset, select_overfit_samples, split_samples
from .losses import ellipse_losses
from .model import EllipseCenterNet, parameter_count
from .runtime import evaluate_predictions, predict_samples, save_curves, save_predictions, save_visualizations


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl_baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


def train(config: dict) -> dict:
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    per_dataset = int(config["train"]["overfit_per_dataset"])
    if per_dataset:
        train_samples = select_overfit_samples(config["datasets"], per_dataset, config["seed"])
        validation_samples = train_samples
        test_samples = train_samples
    else:
        train_samples, validation_samples, test_samples = [], [], []
        for dataset_name in config["datasets"]:
            train, validation, test = split_samples(
                dataset_name,
                config["train"]["validation_fraction"],
                config["train"]["test_fraction"],
                config["seed"],
            )
            train_samples.extend(train)
            validation_samples.extend(validation)
            test_samples.extend(test)
    directory = run_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")
    sample_splits = {
        "train": [{"dataset": sample.dataset, "name": sample.name} for sample in train_samples],
        "validation": [{"dataset": sample.dataset, "name": sample.name} for sample in validation_samples],
        "test": [{"dataset": sample.dataset, "name": sample.name} for sample in test_samples],
    }
    (directory / "samples.json").write_text(json.dumps(sample_splits, indent=2), encoding="utf-8")
    dataset = EllipseDataset(train_samples, config["input_size"], config["output_stride"], config["slots"])
    loader = DataLoader(
        dataset,
        batch_size=min(config["train"]["batch_size"], len(dataset)),
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = EllipseCenterNet(config["slots"], config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
    )
    amp_enabled = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    first_loss = None
    for step in range(1, config["train"]["steps"] + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        tensor_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            losses = ellipse_losses(model(tensor_batch["image"]), tensor_batch, config["loss"])
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite loss at step {step}: {losses}")
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        row = {"step": step, **{name: float(value.detach()) for name, value in losses.items()}}
        history.append(row)
        first_loss = row["total"] if first_loss is None else first_loss
        if step % config["train"]["log_interval"] == 0 or step == 1:
            print(" ".join(f"{key}={value:.6f}" if key != "step" else f"step={value}" for key, value in row.items()))
        if step % config["train"]["eval_interval"] == 0 or step == config["train"]["steps"]:
            predictions = predict_samples(model, validation_samples, config, device)
            metrics = evaluate_predictions(validation_samples, predictions, config["evaluate"]["iou_threshold"])
            score = metrics["standard"]["best_f1"]
            print(f"eval step={step} standard_f1={score:.6f} legacy_f1={metrics['legacy']['f1']:.6f}")
            checkpoint = {
                "model": model.state_dict(),
                "config": {k: v for k, v in config.items() if not k.startswith("_")},
                "step": step,
                "metrics": metrics,
                "sample_splits": sample_splits,
                "selected_threshold": metrics["standard"]["best_threshold"],
            }
            torch.save(checkpoint, directory / "last.pt")
            if score > best_f1:
                best_f1 = score
                torch.save(checkpoint, directory / "best.pt")
                save_predictions(directory / "predictions.json", predictions)
                save_curves(directory / "curves.png", metrics["standard"])
                save_visualizations(
                    directory / "visualizations",
                    validation_samples,
                    predictions,
                    config["evaluate"]["save_visualizations"],
                    metrics["standard"]["best_threshold"],
                )
                (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "parameters": parameter_count(model),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "initial_loss": first_loss,
        "final_loss": history[-1]["total"],
        "loss_reduction": 1.0 - history[-1]["total"] / first_loss,
        "best_standard_f1": best_f1,
    }
    (directory / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
