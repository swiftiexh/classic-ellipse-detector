# Datasets

## Prasad

Location: `datasets/prasad`

This is the public Prasad ellipse detection dataset extracted from the official
AAMED dataset package.

Structure:

```txt
datasets/prasad/
  imagenames.txt
  images/
  gt/
  AAMED/
```

Notes:

- `imagenames.txt` lists 198 images.
- Ground-truth files use the `gt_` prefix, for example
  `gt_002_0038.jpg.txt`.
- The ground-truth format is compatible with `--gt-format plain_rad`.
- `AAMED/` contains the official AAMED result files included in the dataset
  package.

Example evaluation command:

```powershell
.\build\bin\aamed_eval.exe `
  --dataset-root .\datasets\prasad `
  --results-dir .\output\prasad `
  --gt-prefix gt_ `
  --gt-format plain_rad `
  --result-format aamed_fled `
  --overlap 0.8 `
  --report .\output\prasad_eval_result.txt
```
