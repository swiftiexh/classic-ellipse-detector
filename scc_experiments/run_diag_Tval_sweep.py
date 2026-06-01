"""Diagnostic experiment: test different validation thresholds to locate Recall bottleneck.

Runs the fixed-baseline detector with T_val in {0.01, 0.10, 0.30, 0.50, 0.77}
on all 198 Prasad images, then evaluates each with aamed_eval.exe.
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

THRESHOLDS = [0.01, 0.10, 0.30, 0.50, 0.77]


def _detector_env() -> dict:
    """Return os.environ with OpenCV DLL dir prepended to PATH."""
    env = os.environ.copy()
    env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    return env


def run_detector(t_val: float) -> tuple[Path, dict]:
    """Run detector on all Prasad images with given T_val. Returns output dir and stats."""
    output_dir = ROOT / "output" / f"scc_diag_Tval_{t_val:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _detector_env()

    image_names = [
        line.strip()
        for line in IMAGE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results = {"total": len(image_names), "success": 0, "failed": 0, "timeouts": 0,
               "total_detections": 0, "images": {}}
    start = time.time()

    for i, name in enumerate(image_names):
        image_path = IMAGE_DIR / name
        if not image_path.exists():
            print(f"  [{i+1:03d}/{len(image_names)}] MISSING: {name}")
            results["failed"] += 1
            continue

        try:
            proc = subprocess.run(
                [str(EXE), "--input", str(image_path),
                 "--output-dir", str(output_dir),
                 "--T-val", f"{t_val:.2f}", "--quiet"],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"  [{i+1:03d}/{len(image_names)}] TIMEOUT: {name}")
            results["timeouts"] += 1
            continue

        if proc.returncode != 0:
            print(f"  [{i+1:03d}/{len(image_names)}] FAIL({proc.returncode}): {name}")
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
                  f"{results['failed']} failed, {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"  Done: {results['success']}/{results['total']} succeeded, "
          f"{results['failed']} failed, total_dets={results['total_detections']}, "
          f"time={elapsed:.1f}s")

    return output_dir, results


def run_eval(t_val: float, results_dir: Path) -> dict:
    """Run aamed_eval.exe on results directory. Returns parsed metrics."""
    env = _detector_env()
    report_path = ROOT / "output" / f"scc_diag_Tval_{t_val:.2f}_eval.txt"
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
        print(f"ERROR: detector executable not found: {EXE}")
        return 1
    if not EVAL_EXE.exists():
        print(f"ERROR: eval executable not found: {EVAL_EXE}")
        return 1

    all_metrics = {}
    print("=" * 60)
    print("DIAGNOSTIC: Recall Bottleneck Detection via Validation Threshold Sweep")
    print("=" * 60)

    for t_val in THRESHOLDS:
        tag = f"T_val={t_val:.2f}"
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        output_dir, run_stats = run_detector(t_val)
        print(f"  → Total detections across 198 images: {run_stats['total_detections']}")

        metrics = run_eval(t_val, output_dir)
        all_metrics[tag] = metrics
        print(f"  → Eval: P={metrics.get('Precision', '?'):.4f}  "
              f"R={metrics.get('Recall', '?'):.4f}  "
              f"F={metrics.get('FMeasure', '?'):.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Validation Threshold Sweep Results")
    print(f"{'='*60}")
    print(f"{'T_val':<12} {'Precision':>10} {'Recall':>10} {'FMeasure':>10} {'PosMatches':>12} {'Detections':>12}")
    print("-" * 70)

    baseline_recall = all_metrics.get("T_val=0.77", {}).get("Recall", 0.0)
    summary_data = {}

    for t_val in THRESHOLDS:
        tag = f"T_val={t_val:.2f}"
        m = all_metrics.get(tag, {})
        p = m.get("Precision", 0)
        r = m.get("Recall", 0)
        f = m.get("FMeasure", 0)
        pm = int(m.get("PositiveMatches", 0))
        det = int(m.get("DetectedCount", 0))
        delta_r = r - baseline_recall

        print(f"{tag:<12} {p:>10.6f} {r:>10.6f} {f:>10.6f} {pm:>12} {det:>12}")

        summary_data[tag] = {
            "T_val": t_val,
            "Precision": p,
            "Recall": r,
            "FMeasure": f,
            "PositiveMatches": pm,
            "DetectedCount": det,
            "DeltaRecall": delta_r,
        }

    # Diagnostic conclusion
    print(f"\n{'='*60}")
    print("DIAGNOSTIC ANALYSIS")
    print(f"{'='*60}")
    print(f"\nBaseline (T_val=0.77):")
    bl = summary_data["T_val=0.77"]
    t01 = summary_data["T_val=0.01"]
    t01_recall = t01["Recall"]

    print(f"  Precision={bl['Precision']:.6f}, Recall={bl['Recall']:.6f}, "
          f"FMeasure={bl['FMeasure']:.6f}")

    max_recall = t01_recall
    recall_gain = max_recall - baseline_recall
    gt_total = 1165  # Known from Prasad dataset

    print(f"\nWith validation essentially off (T_val=0.01):")
    print(f"  Recall={t01['Recall']:.6f}, Precision={t01['Precision']:.6f}")
    print(f"  Maximum possible Recall gain from improving validation: ΔRecall={recall_gain:.6f}")
    print(f"  Detections: {t01['DetectedCount']}")

    # Interpret
    max_possible_recall_from_val = recall_gain
    remaining_recall_gap = 1.0 - t01_recall
    print(f"\nInterpretation:")
    print(f"  - Recall gain from lowering validation threshold: {max_possible_recall_from_val:.4f}")
    print(f"  - Remaining Recall gap even with validation off: {1.0 - t01_recall:.4f}")

    if max_possible_recall_from_val < 0.02:
        print(f"  - !! CONCLUSION: Recall bottleneck is NOT in validation module.")
        print(f"    Even with validation essentially off, Recall barely improves.")
        print(f"    The bottleneck is upstream: arc segmentation, grouping, or search.")
    elif max_possible_recall_from_val < 0.08:
        print(f"  - !! CONCLUSION: Validation contributes partially to Recall bottleneck.")
        print(f"    Improving validation ALONE will yield modest gains.")
        print(f"    Recommend combining with upstream improvements.")
    else:
        print(f"  - OK CONCLUSION: Validation IS a significant Recall bottleneck.")
        print(f"    Improving the scoring function could yield substantial gains.")

    # Save summary
    summary_path = LOG_DIR / "diag_Tval_sweep_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
