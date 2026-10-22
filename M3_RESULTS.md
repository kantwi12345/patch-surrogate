# M3 — Inverse Design: Validation Results

## Method

`src/invert.py` implements gradient-based inverse design on top of the frozen
M2.5 parametric surrogate (`checkpoints/surrogate_param_best.pt`), rather than
the superseded dense-curve surrogate.

Pipeline:

1. Geometry is parameterized as an unconstrained tensor `theta` passed through
   a `sigmoid` reparameterization into physical bounds (`GEOM_BOUNDS`) for
   `L, W, h, eps_r`. The feed inset `y0` is derived as a fraction of
   `0.48 * L`, guaranteeing it's always physically valid regardless of the
   other sampled params.
2. The normalized geometry is passed through the frozen surrogate to predict
   `[fr_hz, Q, Rin_matched, gain_at_res_dbi]` (denormalized via the exact
   log10 + z-score stats saved in the checkpoint).
3. Loss is computed **directly on the compact predicted params**, not on a
   reconstructed 602-point curve:
   - Frequency term: squared relative error vs. target frequency.
   - Return-loss term: closed-form `min_S11_at_resonance = 20*log10(|(Rin-50)/(Rin+50)|)`,
     penalized only if shallower than the target (deeper-than-requested match
     is free, not penalized).
   - Gain term: either a direct target match (if gain target given) or a mild
     regularizer nudging toward higher gain.
   - This is mathematically equivalent to matching the reconstructed S11 curve
     at resonance for a freq/min-S11/gain spec, since min-S11-at-resonance is
     a closed-form function of `Rin_matched` alone. Avoids needing a
     differentiable version of `reconstruct_curves()`. If a bandwidth target
     is ever needed, a `Q` term should be added to the loss at that point.
4. Multi-start: 40 random initializations, Adam optimizer, 400 steps.
5. Best candidate's `eps_r, h` are snapped to the nearest entry in a small
   catalog of standard, purchasable substrates (`STANDARD_SUBSTRATES`:
   Rogers RO3003/RO4003C/RO3010, RT/duroid 5880, FR4). `L, W` are then
   regenerated via the closed-form `nominal_design()` at the target frequency
   on the snapped substrate, and `y0` is re-solved via `find_matched_inset()`
   for a fresh 50Ω match — rather than literally rounding the raw optimized
   L/W/y0 floats, which would not account for the discrete eps_r/h jump.
6. The snapped design is run through the **real analytical `simulate()`**
   (ground truth, surrogate not involved) for final validation.

## Test Results

Four specs tested, spanning the sampler's trained frequency range (1-6 GHz)
plus a tightened return-loss case:

| Target freq | Target min S11 | True freq (simulate) | Freq error | True min S11 | True peak gain | True BW |
|---|---|---|---|---|---|---|
| 2.4 GHz | ≤ -15 dB | 2.402 GHz | 2 MHz | -23.85 dB | 5.37 dBi | 0 MHz* |
| 5.0 GHz | ≤ -15 dB | 4.987 GHz | 13 MHz | -20.94 dB | 4.65 dBi | 75 MHz |
| 3.5 GHz | ≤ -20 dB | 3.506 GHz | 6 MHz | -20.42 dB | 5.13 dBi | 25 MHz |
| 1.2 GHz | ≤ -15 dB | 1.201 GHz | 1 MHz | -21.72 dB | 5.78 dBi | 0 MHz* |

All 4 cases: frequency error well under the 50 MHz median threshold
established for surrogate accuracy (max observed: 13 MHz), and measured
return loss always met or exceeded the requested spec, including the
tightened -20 dB case (met with ~0.4 dB margin, reflecting a genuinely harder
target near the edge of what the geometry/substrate choice allows).

Example resulting design (2.4 GHz, ≤-15dB spec):