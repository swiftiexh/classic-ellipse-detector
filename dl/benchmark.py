from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .config import load_config
from .model import EllipseCenterNet, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl_baseline.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = EllipseCenterNet(config["slots"], False).to(device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    image = torch.randn(1, 3, config["input_size"], config["input_size"], device=device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.inference_mode():
        for _ in range(config["benchmark"]["warmup"]):
            model(image)
        synchronize()
        durations = []
        for _ in range(config["benchmark"]["iterations"]):
            start = time.perf_counter()
            model(image)
            synchronize()
            durations.append((time.perf_counter() - start) * 1000.0)
    result = {
        "device": str(device),
        "parameters": parameter_count(model),
        "mean_ms": float(np.mean(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "fps": 1000.0 / float(np.mean(durations)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
