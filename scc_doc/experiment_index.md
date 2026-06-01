# Experiment Index

Generated: 2026-05-30

This directory has been split by experiment round so the two changes are no longer mixed at the top level.

Chinese summary:

```text
scc_doc/总结.md
```

## Current Answer

The current code/workspace is after **Round 02: Fixed Baseline and SCC-Enhance**.

The important current code change is:

```text
src/Group.cpp
```

It contains the `isCombValid` scope fix in `BiDirectionVerification`. This is the recommended change to keep.

The current final summary is:

```text
scc_doc/round_02_fixed_enhance/05_consolidated_final_report.md
```

## Directory Map

```text
scc_doc/
  experiment_index.md
  总结.md
  common/
    00_project_survey.md
    01_baseline_result.md
  round_01_geofilter/
    03_full_experiment.md
    04_final_report.md
  round_02_fixed_enhance/
    02_idea_selection.md
    03_sanity_check.md
    04_full_experiment.md
    05_consolidated_final_report.md
  attempts/
    01_scc_geofilter.md
    02_scc_enhance.md
  logs/
    common/
    round_01_geofilter/
    round_02_fixed_enhance/
  figures/
  tables/
  prompts/
```

## Round 00: Common Baseline

Shared project inspection and baseline evaluation:

| File | Purpose |
| --- | --- |
| `scc_doc/common/00_project_survey.md` | Project structure, algorithm flow, dataset and evaluator notes |
| `scc_doc/common/01_baseline_result.md` | Single-image baseline and official Prasad AAMED evaluation |

Official AAMED baseline:

| Precision | Recall | FMeasure | AverageDetectedTimeMs |
| ---: | ---: | ---: | ---: |
| 0.771285 | 0.396567 | 0.523810 | 5.146824 |

## Round 01: SCC-GeoFilter

This round added a post-processing module over existing `.fled.txt` outputs. It did **not** modify the C++ detector.

| File | Purpose |
| --- | --- |
| `scc_doc/round_01_geofilter/03_full_experiment.md` | Full GeoFilter experiment record |
| `scc_doc/round_01_geofilter/04_final_report.md` | GeoFilter final report |
| `scc_doc/attempts/01_scc_geofilter.md` | Short attempt summary |

Best result:

| Method | Precision | Recall | FMeasure | AverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| SCC-GeoFilter strict | 0.776094 | 0.395708 | 0.524161 | 5.146824 |

Conclusion: keep as a reference direction, but next step should be soft scoring / re-ranking instead of stricter hard filtering.

## Round 02: Fixed Baseline and SCC-Enhance

This round contains two parts:

1. `src/Group.cpp` `isCombValid` scope fix.
2. SCC-Enhance image preprocessing experiment using CLAHE + bilateral filtering.

| File | Purpose |
| --- | --- |
| `scc_doc/round_02_fixed_enhance/02_idea_selection.md` | Idea selection for the second round |
| `scc_doc/round_02_fixed_enhance/03_sanity_check.md` | SCC-Enhance sanity check |
| `scc_doc/round_02_fixed_enhance/04_full_experiment.md` | Full SCC-Enhance experiment |
| `scc_doc/round_02_fixed_enhance/05_consolidated_final_report.md` | Current consolidated final report |
| `scc_doc/attempts/02_scc_enhance.md` | Short attempt summary |

Fixed detector baseline:

| Method | Precision | Recall | FMeasure | AverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| Fixed detector baseline | 0.779070 | 0.402575 | 0.530843 | 14.952466 |

SCC-Enhance did not beat the fixed detector baseline:

| Method | Precision | Recall | FMeasure | AverageDetectedTimeMs |
| --- | ---: | ---: | ---: | ---: |
| SCC-Enhance light | 0.728296 | 0.388841 | 0.506995 | 15.036036 |
| SCC-Enhance medium | 0.727119 | 0.368240 | 0.488889 | 17.242711 |
| SCC-Enhance strong | 0.783985 | 0.361373 | 0.494712 | 15.353225 |

Conclusion: keep `Group.cpp` bugfix; do not continue the current hard-coded SCC-Enhance route.

## Recommended Mainline

Keep:

```text
src/Group.cpp isCombValid bugfix
```

Optional reference:

```text
SCC-GeoFilter strict
```

Do not treat as current best:

```text
SCC-Enhance CLAHE + bilateral preprocessing
```

Suggested next experiment:

```text
SCC-GeoScore: candidate-level soft geometry scoring / confidence calibration
```
