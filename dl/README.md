# Lightweight CenterNet Ellipse Baseline

This package is an independent PyTorch baseline for comparison with the
traditional AAMED implementation. It does not change the C++ detector.

## Architecture

- MobileNetV3-Small single-scale backbone.
- Lightweight deconvolution decoder with output stride 4.
- Five instance slots per output cell, ordered by ellipse area inside each
  cell. This preserves all targets in the three real datasets, including cells
  containing several ellipse centers.
- Each slot predicts a center heatmap, center offset, log semi-axes, and
  `[cos(2 theta), sin(2 theta)]`.
- No FPN, multi-scale inference, ellipse NMS, or innovation module is enabled
  in the baseline.

The extension points for later ablations are `EllipseCenterNet.neck`,
`EllipseCenterNet.quality_refiner`, and the decoder/post-processing functions.

## Commands

Run from the repository root with the CUDA-enabled `d2l_env`:

```powershell
conda run -n d2l_env python -m unittest dl.tests -v
conda run -n d2l_env python -m dl.train --config configs/dl_overfit.yaml
conda run -n d2l_env python -m dl.benchmark --config configs/dl_overfit.yaml --checkpoint output/dl/overfit_12/best.pt
```

Formal training uses deterministic train/validation/test partitions. The
confidence threshold is selected on validation and stored in the checkpoint.
Evaluate a test partition without tuning on it:

```powershell
conda run -n d2l_env python -m dl.evaluate `
  --config configs/dl_baseline.yaml `
  --checkpoint output/dl/baseline/best.pt `
  --checkpoint-split test `
  --output output/dl/baseline/test
```

Evaluate an existing traditional detector output with the Python legacy
evaluator:

```powershell
conda run -n d2l_env python -m dl.evaluate `
  --config configs/dl_baseline.yaml `
  --dataset smartphone `
  --fled-dir output/smartphone/baseline `
  --output output/dl/legacy_check
```

## Evaluation Policy

- `legacy`: reproduces the repository's C++ many-to-many overlap matching for
  direct comparison with existing Precision, Recall, and FMeasure tables.
- `standard`: globally score-ranked one-to-one matching with PR, F1, and AP
  curves.
- Overfit mode deliberately selects its threshold on the same 12 images.
  Formal test evaluation always uses the validation-selected checkpoint
  threshold.

## Verified Overfit Result

The fixed-seed mixed subset contains four non-empty images from each of
Prasad, Random, and Smartphone, with 51 ellipses total.

| Item | Result |
|---|---:|
| Parameters | 1,708,611 |
| Initial loss | 23.805393 |
| Final loss | 0.0000269 |
| Loss reduction | 99.9999% |
| Standard best F1 | 0.969697 |
| Legacy F1 at selected threshold | 0.969697 |
| RTX 4060 mean model forward | about 6.05 ms |

Artifacts are written under `output/dl/overfit_12/`, including checkpoints,
metrics, curves, predictions, history, and twelve overlay visualizations.
