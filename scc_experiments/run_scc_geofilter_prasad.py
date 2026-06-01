from __future__ import annotations

import json
import time
from pathlib import Path

from scc_geofilter import GeoFilterParams, run_geofilter


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "prasad"
IMAGE_NAMES_PATH = DATASET_ROOT / "imagenames.txt"
BASELINE_DIR = DATASET_ROOT / "AAMED"
IMAGE_DIR = DATASET_ROOT / "images"
OUTPUT_ROOT = ROOT / "output"
LOG_DIR = ROOT / "scc_doc" / "logs"


PARAM_SETS = [
    GeoFilterParams(
        name="loose",
        max_axis_ratio=10.0,
        min_area_ratio=0.0002,
        max_area_ratio=1.50,
        center_margin_ratio=0.10,
        bbox_margin_ratio=0.35,
    ),
    GeoFilterParams(
        name="medium",
        max_axis_ratio=6.0,
        min_area_ratio=0.0005,
        max_area_ratio=1.10,
        center_margin_ratio=0.05,
        bbox_margin_ratio=0.20,
    ),
    GeoFilterParams(
        name="strict",
        max_axis_ratio=4.0,
        min_area_ratio=0.0010,
        max_area_ratio=0.85,
        center_margin_ratio=0.00,
        bbox_margin_ratio=0.10,
    ),
]


def main() -> int:
    image_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    started = time.perf_counter()
    for params in PARAM_SETS:
        output_dir = OUTPUT_ROOT / f"scc_geofilter_prasad_{params.name}"
        summary = run_geofilter(image_names, BASELINE_DIR, IMAGE_DIR, output_dir, params)
        all_summaries[params.name] = summary
        print(
            f"{params.name}: images={summary['images']} "
            f"before={summary['totals']['before']} after={summary['totals']['after']} "
            f"removed={summary['totals']['removed']}"
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "elapsed_ms": elapsed_ms,
        "baseline_dir": str(BASELINE_DIR),
        "image_dir": str(IMAGE_DIR),
        "image_names": str(IMAGE_NAMES_PATH),
        "summaries": all_summaries,
    }
    (LOG_DIR / "scc_geofilter_prasad_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"elapsed_ms={elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
