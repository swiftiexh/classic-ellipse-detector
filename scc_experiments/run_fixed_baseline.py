"""Run the fixed (isCombValid bugfix) detector on all Prasad images."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "cmake_build" / "bin" / "Release" / "aamed_demo.exe"
IMAGE_NAMES_PATH = ROOT / "datasets" / "prasad" / "imagenames.txt"
IMAGE_DIR = ROOT / "datasets" / "prasad" / "images"
OUTPUT_DIR = ROOT / "output" / "scc_fixed_baseline"
LOG_DIR = ROOT / "scc_doc" / "logs"


def main() -> int:
    image_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = {"total": len(image_names), "success": 0, "failed": 0, "images": {}}
    start = time.time()

    for i, name in enumerate(image_names):
        image_path = IMAGE_DIR / name
        if not image_path.exists():
            print(f"[{i+1:03d}/{len(image_names)}] MISSING: {name}")
            results["failed"] += 1
            continue

        try:
            proc = subprocess.run(
                [str(EXE), "--input", str(image_path), "--output-dir", str(OUTPUT_DIR)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"[{i+1:03d}/{len(image_names)}] TIMEOUT: {name}")
            results["failed"] += 1
            continue

        if proc.returncode != 0:
            print(f"[{i+1:03d}/{len(image_names)}] FAIL({proc.returncode}): {name}")
            print(f"  stderr: {proc.stderr[:200]}")
            results["failed"] += 1
            continue

        # Copy fled.txt to output dir with proper name
        fled_src = OUTPUT_DIR / f"{name}.fled.txt"
        results["images"][name] = {
            "status": "ok",
            "detections": 0,
        }

        if fled_src.exists():
            lines = fled_src.read_text(encoding="utf-8").splitlines()
            results["images"][name]["detections"] = max(0, len(lines) - 1)

        results["success"] += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(
                f"[{i+1:03d}/{len(image_names)}] processed, "
                f"{results['success']} ok, {results['failed']} failed, "
                f"{elapsed:.1f}s elapsed"
            )

    elapsed = time.time() - start
    total_dets = sum(
        v.get("detections", 0) for v in results["images"].values()
    )
    print(
        f"\nDone: {results['success']}/{results['total']} succeeded, "
        f"{results['failed']} failed, "
        f"total detections={total_dets}, "
        f"time={elapsed:.1f}s"
    )

    (LOG_DIR / "scc_fixed_baseline_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
