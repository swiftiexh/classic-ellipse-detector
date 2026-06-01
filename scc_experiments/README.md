# SCC Experiments

This directory contains two independent experiment attempts.

## Attempt 01: SCC-GeoFilter

Post-processes existing AAMED `.fled.txt` files with lightweight geometry filters.

Scripts:

```text
scc_geofilter.py
run_scc_geofilter_prasad.py
run_scc_geofilter_sanity.py
visualize_scc_cases.ps1
```

Main outputs:

```text
output/scc_geofilter_prasad_loose
output/scc_geofilter_prasad_medium
output/scc_geofilter_prasad_strict
```

## Attempt 02: Fixed Baseline and SCC-Enhance

Uses a small `src/Group.cpp` bugfix, reruns the detector, and tests CLAHE + bilateral preprocessing.

Scripts:

```text
run_fixed_baseline.py
scc_enhance.py
run_scc_enhance.py
run_scc_enhance_sanity.py
```

Main outputs:

```text
output/scc_fixed_baseline
output/scc_enhance_light_results
output/scc_enhance_medium_results
output/scc_enhance_strong_results
```

Generated enhanced image directories are intermediate products and are not needed for final comparison.
