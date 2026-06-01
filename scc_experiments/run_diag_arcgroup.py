"""Diagnostic: Arc Grouping Constraint Relaxation.

Sweeps kd_radius_mul and region_bypass to identify where arc pairs
are being rejected in the grouping/search pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "bin" / "aamed_demo.exe"
EVAL_EXE = ROOT / "build" / "bin" / "aamed_eval.exe"
OPENCV_BIN = Path(r"E:\OPENCV\opencv\build\x64\vc16\bin")
IMAGE_NAMES_PATH = ROOT / "datasets" / "prasad" / "imagenames.txt"
IMAGE_DIR = ROOT / "datasets" / "prasad" / "images"
LOG_DIR = ROOT / "scc_doc" / "logs"

# Test configurations
CONFIGS = [
    # (tag, kd_mul, region_bypass, T_val)
    ("baseline",        1.0, 0, 0.77),
    ("kd=2",            2.0, 0, 0.77),
    ("kd=3",            3.0, 0, 0.77),
    ("kd=5",            5.0, 0, 0.77),
    ("kd=10",          10.0, 0, 0.77),
    ("rgn=1",           1.0, 1, 0.77),
    ("rgn=2",           1.0, 2, 0.77),
    ("kd=3_rgn=2",      3.0, 2, 0.77),
    ("kd=5_rgn=2",      5.0, 2, 0.77),
    ("max_relax",      10.0, 2, 0.50),  # also lower T_val
]


def env_with_opencv() -> dict:
    env = os.environ.copy()
    env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    return env


def run_detector(tag: str, kd_mul: float, region_bypass: int, t_val: float) -> tuple[Path, dict]:
    output_dir = ROOT / "output" / f"scc_diag_arcgroup_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = env_with_opencv()

    image_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results = {"total": len(image_names), "success": 0, "failed": 0,
               "total_detections": 0, "images": {}}
    start = time.time()

    for i, name in enumerate(image_names):
        image_path = IMAGE_DIR / name
        if not image_path.exists():
            results["failed"] += 1
            continue

        try:
            proc = subprocess.run(
                [str(EXE), "--input", str(image_path),
                 "--output-dir", str(output_dir),
                 "--T-val", f"{t_val:.2f}",
                 "--kd-mul", f"{kd_mul:.1f}",
                 "--region-bypass", str(region_bypass),
                 "--quiet"],
                capture_output=True, text=True, timeout=120, env=env,
            )
        except subprocess.TimeoutExpired:
            results["failed"] += 1
            continue

        if proc.returncode != 0:
            results["failed"] += 1
            continue

        fled_src = output_dir / f"{name}.fled.txt"
        detections = 0
        if fled_src.exists():
            lines = fled_src.read_text(encoding="utf-8").splitlines()
            detections = max(0, len(lines) - 1)

        results["images"][name] = {"status": "ok", "detections": detections}
        results["total_detections"] += detections
        results["success"] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:03d}/{len(image_names)}] {results['success']} ok, "
                  f"{elapsed:.1f}s, dets={results['total_detections']}")

    elapsed = time.time() - start
    print(f"  Done: {results['success']}/{results['total']} ok, "
          f"dets={results['total_detections']}, time={elapsed:.1f}s")
    return output_dir, results


def run_eval(tag: str, results_dir: Path) -> dict:
    env = env_with_opencv()
    report_path = ROOT / "output" / f"scc_diag_arcgroup_{tag}_eval.txt"
    cmd = [
        str(EVAL_EXE),
        "--dataset-root", str(ROOT / "datasets" / "prasad"),
        "--results-dir", str(results_dir),
        "--gt-prefix", "gt_",
        "--gt-format", "plain_rad",
        "--result-format", "aamed_fled",
        "--overlap", "0.8",
        "--report", str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    metrics = {}
    for line in proc.stdout.splitlines():
        if ":" in line and not line.startswith(" "):
            key, val = line.split(":", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                metrics[key.strip()] = val.strip()
    return metrics


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not EXE.exists():
        print(f"ERROR: {EXE} not found")
        return 1

    all_metrics = {}
    print("=" * 70)
    print("DIAGNOSTIC: Arc Grouping Constraint Relaxation")
    print("=" * 70)

    for tag, kd_mul, rgn, t_val in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Config: {tag} (kd_mul={kd_mul}, region_bypass={rgn}, T_val={t_val})")
        print(f"{'='*70}")

        output_dir, run_stats = run_detector(tag, kd_mul, rgn, t_val)
        metrics = run_eval(tag, output_dir)
        all_metrics[tag] = {**metrics, "kd_mul": kd_mul, "region_bypass": rgn,
                            "T_val": t_val, "total_detections": run_stats["total_detections"]}
        print(f"  -> P={metrics.get('Precision', 0):.4f}  "
              f"R={metrics.get('Recall', 0):.4f}  "
              f"F={metrics.get('FMeasure', 0):.4f}  "
              f"PM={metrics.get('PositiveMatches', 0):.0f}  "
              f"Dets={run_stats['total_detections']}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Arc Grouping Constraint Relaxation")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'kd_mul':>7} {'rgn':>4} {'T_val':>6} {'Precision':>10} {'Recall':>10} {'FMeasure':>10} {'PosMatch':>9} {'Dets':>7}")
    print("-" * 95)

    baseline_r = all_metrics.get("baseline", {}).get("Recall", 0)
    baseline_f = all_metrics.get("baseline", {}).get("FMeasure", 0)

    for tag, _, _, _ in CONFIGS:
        m = all_metrics.get(tag, {})
        print(f"{tag:<18} {m.get('kd_mul',0):>7.1f} {m.get('region_bypass',0):>4} "
              f"{m.get('T_val',0):>6.2f} {m.get('Precision',0):>10.6f} "
              f"{m.get('Recall',0):>10.6f} {m.get('FMeasure',0):>10.6f} "
              f"{int(m.get('PositiveMatches',0)):>9} {int(m.get('total_detections',0)):>7}")

    best = max(all_metrics.items(), key=lambda x: x[1].get("FMeasure", 0))
    print(f"\nBest config by FMeasure: {best[0]} (F={best[1].get('FMeasure', 0):.6f})")

    # Bottleneck analysis
    max_recall_config = max(all_metrics.items(), key=lambda x: x[1].get("Recall", 0))
    max_recall = max_recall_config[1].get("Recall", 0)
    recall_gain = max_recall - baseline_r

    print(f"\n{'='*70}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*70}")
    print(f"Baseline Recall: {baseline_r:.6f} (F={baseline_f:.6f})")
    print(f"Max Recall (with aggressive relaxation): {max_recall:.6f} "
          f"({max_recall_config[0]})")
    print(f"Delta Recall from arc grouping relaxation: +{recall_gain:.6f}")
    print(f"Remaining Recall gap (to 1.0): {1.0 - max_recall:.6f}")
    print()

    if recall_gain < 0.02:
        print("CONCLUSION: Arc grouping constraints are NOT the main Recall bottleneck.")
        print("  Even with aggressive constraint relaxation (kd_mul up to 10,")
        print("  region_bypass=2, lower T_val), Recall barely moves.")
        print("  The bottleneck is most likely in: ARC SEGMENTATION (FSA step)")
        print("  or EDGE EXTRACTION (Canny + contour finding).")
    elif recall_gain < 0.06:
        print("CONCLUSION: Arc grouping contributes partially to the Recall bottleneck.")
        print("  Constraint relaxation helps but doesn't fundamentally change the picture.")
    else:
        print("CONCLUSION: Arc grouping constraints ARE a significant bottleneck!")
        print("  Relaxing constraints yields substantial Recall improvement.")

    # Save
    summary_path = LOG_DIR / "diag_arcgroup_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
