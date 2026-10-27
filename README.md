# Patch Antenna Design Surrogate

A neural-network-based inverse design tool for rectangular microstrip patch
antennas: specify a target frequency band and match quality, get a
manufacturable geometry back in milliseconds, validated against both an
analytical cavity model and full-wave HFSS simulation.

## Pipeline Overview

1. **Analytical forward model** (`src/sim/analytical_model.py`): cavity-model
   + two-slot radiating-aperture physics giving S11(f) and gain(f) from
   geometry (L, W, h, eps_r, y0) in microseconds. Used to generate training
   data and as a fast self-consistency check.
2. **Synthetic dataset** (LHS sampling across the design space) + **surrogate
   MLP** (`src/train_surrogate.py` or similar, PyTorch) trained to predict
   S11/gain curves directly from geometry, for near-instant forward
   evaluation during inversion.
3. **Inversion** (`src/invert.py`): given a target frequency and spec
   (e.g. min S11 <= -15dB), gradient-optimizes geometry against the
   surrogate from many random restarts, then snaps the result to a
   manufacturable substrate (FR4, RO4003C, RO3003, RO3010/Alumina,
   RT/duroid5880, etc.) and re-solves the feed inset depth analytically.
4. **HFSS validation** (`src/sim/hfss_model.py`, via pyaedt + AEDT 2022 R1):
   full-wave simulation of specific designs for ground-truth comparison
   against the analytical model.

## Key Finding (M4/M5)

The analytical model's resonant-frequency prediction is reasonably accurate
(~5% error vs. HFSS, bounded and systematic), but its impedance-matching
formula (inset feed depth `y0` for 50-ohm match) was found to be severely
wrong in 100% of tested designs (min S11 as poor as -0.1dB in HFSS despite
predicting <=-20dB). Root-caused and fixed in M5 with an HFSS-calibrated
empirical correction factor (`Y0_HFSS_CORRECTION = 0.72` in
`find_matched_inset()`), validated to bring all 8 test designs to HFSS
min S11 <= -11dB (7/8 <= -15dB). See `M4_RESULTS.md` and `M5_RESULTS.md`
for full details.

## Usage

```bash
python -m src.invert --freq_ghz 2.4 --min_s11_db -15.0 [--gain_dbi G]