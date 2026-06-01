"""Run SCC-Enhance preprocessing + detector on all Prasad images."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from scc_enhance import ALL_PARAMS, EnhanceParams, enhance_and_save

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "cmake_build" / "bin" / "Release" / "aamed_demo.exe"
IMAGE_NAMES_PATH = ROOT / "datasets" / "prasad" / "imagenames.txt"
IMAGE_DIR = ROOT / "datasets" / "prasad" / "images"
OUTPUT_BASE = ROOT / "output"
LOG_DIR = ROOT / "scc_doc" / "logs"


def process_image(
    image_name: str,
    params: EnhanceParams,
    enhanced_dir: Path,
    results_dir: Path,
) -> dict:
    """Enhance a single image and run detector on it. Returns stats dict."""
    src = IMAGE_DIR / image_name
    enhanced_path = enhanced_dir / image_name

    try:
        enhance_and_save(src, enhanced_path, params)
    except Exception as e:
        return {"status": "enhance_failed", "error": str(e)}

    try:
        proc = subprocess.run(
            [str(EXE), "--input", str(enhanced_path), "--output-dir", str(results_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    if proc.returncode != 0:
        return {"status": "detect_failed", "returncode": proc.returncode, "stderr": proc.stderr[:500]}

    fled_src = results_dir / f"{image_name}.fled.txt"
    detections = 0
    if fled_src.exists():
        lines = fled_src.read_text(encoding="utf-8").splitlines()
        detections = max(0, len(lines) - 1)

    return {"status": "ok", "detections": detections}


def run_experiment(params: EnhanceParams) -> dict:
    """Run full experiment with given params on all images."""
    image_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    param_key = params.name
    enhanced_dir = OUTPUT_BASE / f"scc_enhance_{param_key}_images"
    results_dir = OUTPUT_BASE / f"scc_enhance_{param_key}_results"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "params": {
            "name": params.name,
            "clahe_clip_limit": params.clahe_clip_limit,
            "clahe_tile_grid": list(params.clahe_tile_grid),
            "bilateral_d": params.bilateral_d,
            "bilateral_sigma_color": params.bilateral_sigma_color,
            "bilateral_sigma_space": params.bilateral_sigma_space,
        },
        "total": len(image_names),
        "success": 0,
        "failed": 0,
        "total_detections": 0,
        "images": {},
    }

    start = time.time()
    for i, name in enumerate(image_names):
        result = process_image(name, params, enhanced_dir, results_dir)
        summary["images"][name] = result

        if result["status"] == "ok":
            summary["success"] += 1
            summary["total_detections"] += result.get("detections", 0)
        else:
            summary["failed"] += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(
                f"[{param_key}] [{i+1:03d}/{len(image_names)}] "
                f"{summary['success']} ok, {summary['failed']} failed, "
                f"{summary['total_detections']} dets, {elapsed:.1f}s"
            )

    elapsed = time.time() - start
    print(
        f"[{param_key}] Done: {summary['success']}/{summary['total']} succeeded, "
        f"{summary['failed']} failed, total_detections={summary['total_detections']}, "
        f"time={elapsed:.1f}s"
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"scc_enhance_{param_key}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


def main() -> int:
    all_summaries = {}
    for params in ALL_PARAMS:
        print(f"\n{'='*60}")
        print(f"Running SCC-Enhance [{params.name}]...")
        print(f"{'='*60}")
        all_summaries[params.name] = run_experiment(params)

    total = sum(s["total_detections"] for s in all_summaries.values())
    ok = sum(s["success"] for s in all_summaries.values())
    print(f"\nAll experiments complete. Total detections across all params: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
