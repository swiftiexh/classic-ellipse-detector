"""Sanity check: run SCC-Enhance light on first 10 Prasad images."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scc_enhance import LIGHT, EnhanceParams, enhance_and_save

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "cmake_build" / "bin" / "Release" / "aamed_demo.exe"
IMAGE_NAMES_PATH = ROOT / "datasets" / "prasad" / "imagenames.txt"
IMAGE_DIR = ROOT / "datasets" / "prasad" / "images"
OUTPUT_DIR = ROOT / "output" / "scc_enhance_sanity"
LOG_DIR = ROOT / "scc_doc" / "logs"


def main() -> int:
    all_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_names = all_names[:10]

    enhanced_dir = OUTPUT_DIR / "images"
    results_dir = OUTPUT_DIR / "results"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    (results_dir / "imagenames.txt").write_text(
        "\n".join(image_names) + "\n", encoding="utf-8"
    )

    summary = {
        "params": {
            "name": LIGHT.name,
            "clahe_clip_limit": LIGHT.clahe_clip_limit,
            "clahe_tile_grid": list(LIGHT.clahe_tile_grid),
            "bilateral_d": LIGHT.bilateral_d,
            "bilateral_sigma_color": LIGHT.bilateral_sigma_color,
            "bilateral_sigma_space": LIGHT.bilateral_sigma_space,
        },
        "total": len(image_names),
        "success": 0,
        "failed": 0,
        "total_detections": 0,
        "images": {},
    }

    for i, name in enumerate(image_names):
        src = IMAGE_DIR / name
        enhanced_path = enhanced_dir / name
        if not src.exists():
            summary["images"][name] = {"status": "missing_source"}
            summary["failed"] += 1
            print(f"[{i+1:02d}/{len(image_names)}] MISSING: {name}")
            continue

        try:
            enhance_and_save(src, enhanced_path, LIGHT)
        except Exception as e:
            summary["images"][name] = {"status": "enhance_failed", "error": str(e)}
            summary["failed"] += 1
            print(f"[{i+1:02d}/{len(image_names)}] ENHANCE FAIL: {name} - {e}")
            continue

        try:
            proc = subprocess.run(
                [str(EXE), "--input", str(enhanced_path), "--output-dir", str(results_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            summary["images"][name] = {"status": "timeout"}
            summary["failed"] += 1
            print(f"[{i+1:02d}/{len(image_names)}] TIMEOUT: {name}")
            continue

        if proc.returncode != 0:
            summary["images"][name] = {
                "status": "detect_failed",
                "returncode": proc.returncode,
                "stderr": proc.stderr[:300],
            }
            summary["failed"] += 1
            print(f"[{i+1:02d}/{len(image_names)}] DETECT FAIL({proc.returncode}): {name}")
            continue

        fled_src = results_dir / f"{name}.fled.txt"
        detections = 0
        if fled_src.exists():
            lines = fled_src.read_text(encoding="utf-8").splitlines()
            detections = max(0, len(lines) - 1)

        summary["images"][name] = {"status": "ok", "detections": detections}
        summary["success"] += 1
        summary["total_detections"] += detections
        print(f"[{i+1:02d}/{len(image_names)}] {name}: {detections} detections")

    (LOG_DIR / "scc_enhance_sanity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"\nsanity_enhance_light: images={summary['success']}/{summary['total']} "
        f"failed={summary['failed']} total_detections={summary['total_detections']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
