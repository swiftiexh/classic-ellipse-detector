from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 3407,
    "device": "cuda",
    "input_size": 512,
    "output_stride": 4,
    "slots": 5,
    "topk": 200,
    "score_threshold": 0.1,
    "datasets": ["prasad", "random", "smartphone"],
    "model": {"pretrained": False},
    "loss": {"heatmap": 1.0, "offset": 1.0, "axes": 0.5, "angle": 0.2},
    "train": {
        "run_name": "baseline",
        "batch_size": 8,
        "steps": 2000,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "amp": True,
        "num_workers": 0,
        "log_interval": 20,
        "eval_interval": 200,
        "overfit_per_dataset": 0,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "evaluate": {"iou_threshold": 0.8, "save_visualizations": 12},
    "benchmark": {"warmup": 30, "iterations": 100},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, user)
    config["_config_path"] = str(path.resolve())
    return config


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_dir(config: dict[str, Any]) -> Path:
    return project_root() / "output" / "dl" / config["train"]["run_name"]
