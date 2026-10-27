# M5 Results: HFSS-Calibrated Inset-Depth Correction

## Problem
M4 found the analytical model's feed-matching formula (`find_matched_inset`)
systematically overshoots the true HFSS-matched inset depth y0, causing
severe impedance mismatch (min S11 as poor as -0.1dB) in 100% of tested
designs, despite reasonable resonant-frequency accuracy.

## Method
Golden-section search (6 evaluations/geometry) directly on HFSS min S11
(reusing the validated `simulate_hfss()` pipeline) found the true matched
y0 for the same 8 geometries used in M4 validation (1.2-5.8 GHz, 5
substrates). Results (`data/m5_y0_bisect.json`):

| Geometry | y0_analytical | y0_true | ratio |
|---|---|---|---|
| 1.2GHz FR4 | 22.011mm | 16.806mm | 0.764 |
| 1.8GHz FR4 | 14.619mm | 10.784mm | 0.738 |
| 2.4GHz FR4 | 10.905mm | 7.589mm | 0.696 |
| 3.0GHz Alumina | 6.270mm | 5.635mm | 0.899 |
| 3.5GHz RO4003C | 8.052mm | 5.267mm | 0.654 |
| 4.2GHz RO3003 | 7.151mm | 4.678mm | 0.654 |
| 5.0GHz RO3003 | 5.967mm | 4.306mm | 0.722 |
| 5.8GHz RT/duroid5880 | 5.958mm | 3.897mm | 0.654 |

Ratios cluster tightly (0.654-0.899, mean 0.723) with no clean trend vs.
eps_r given only 8 points, so a single global correction factor was applied
rather than risk overfitting a per-parameter model:

```python
Y0_HFSS_CORRECTION = 0.72
y0_corrected = y0_analytical * Y0_HFSS_CORRECTION