from __future__ import annotations

import json
from pathlib import Path

from scc_geofilter import GeoFilterParams, run_geofilter


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "prasad"
IMAGE_NAMES_PATH = DATASET_ROOT / "imagenames.txt"
BASELINE_DIR = DATASET_ROOT / "AAMED"
IMAGE_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = ROOT / "output" / "scc_geofilter_sanity_strict"
LOG_DIR = ROOT / "scc_doc" / "logs"


STRICT_PARAMS = GeoFilterParams(
    name="sanity_strict",
    max_axis_ratio=4.0,
    min_area_ratio=0.0010,
    max_area_ratio=0.85,
    center_margin_ratio=0.00,
    bbox_margin_ratio=0.10,
)


def main() -> int:
    all_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_names = all_names[:10]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "imagenames.txt").write_text(
        "\n".join(image_names) + "\n",
        encoding="utf-8",
    )

    summary = run_geofilter(image_names, BASELINE_DIR, IMAGE_DIR, OUTPUT_DIR, STRICT_PARAMS)
    payload = {
        "sample_count": len(image_names),
        "image_names": image_names,
        "output_dir": str(OUTPUT_DIR),
        "summary": summary,
    }
    (LOG_DIR / "scc_geofilter_sanity_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"sanity_strict: images={summary['images']} "
        f"before={summary['totals']['before']} after={summary['totals']['after']} "
        f"removed={summary['totals']['removed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
