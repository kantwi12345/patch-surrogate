"""
M5 (revised): find the TRUE matched y0 per geometry via golden-section
search on min_s11_db, reusing the already-validated simulate_hfss() pipeline
(full S11 sweep + robust local-minimum resonance detection) instead of
noisy narrow-band Zin extraction.
"""
import json
import numpy as np

from src.sim.schema import DesignParams
from src.sim.hfss_model import simulate_hfss
from src.sim.analytical_model import find_matched_inset, _resonant_frequency

GEOMETRIES = [
    dict(name="1.2GHz_FR4",    L=0.05930, W=0.07600, h=0.00160,  eps_r=4.40),
    dict(name="1.8GHz_FR4",    L=0.03940, W=0.05070, h=0.00160,  eps_r=4.40),
    dict(name="2.4GHz_FR4",    L=0.02940, W=0.03800, h=0.00160,  eps_r=4.40),
    dict(name="3.0GHz_Alum",   L=0.01550, W=0.02110, h=0.00127,  eps_r=10.20),
    dict(name="3.5GHz_RO4003", L=0.02220, W=0.02840, h=0.001524, eps_r=3.55),
    dict(name="4.2GHz_RO3003", L=0.02000, W=0.02520, h=0.001524, eps_r=3.00),
    dict(name="5.0GHz_RO3003", L=0.01670, W=0.02120, h=0.001524, eps_r=3.00),
    dict(name="5.8GHz_RTD",    L=0.01700, W=0.02040, h=0.000787, eps_r=2.20),
]

GOLDEN = (np.sqrt(5) - 1) / 2  # ~0.618
N_ITERS = 4  # total evaluations = 2 + N_ITERS


def eval_s11(L, W, h, eps_r, y0):
    design = DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=float(y0))
    fr_est, _ = _resonant_frequency(L, W, h, eps_r)
    freq_hz = np.linspace(fr_est * 0.85, fr_est * 1.15, 81)
    result = simulate_hfss(design, freq_hz, non_graphical=True, n_points=81)
    print(f"  y0={y0*1e3:.3f}mm -> min_s11={result.min_s11_db:.2f}dB at {result.resonant_freq_hz/1e9:.3f}GHz")
    return result.min_s11_db


results = []
for geo in GEOMETRIES:
    L, W, h, eps_r = geo["L"], geo["W"], geo["h"], geo["eps_r"]
    design = DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=0.0)
    y0_analytical = find_matched_inset(design)

    # Bracket: true value has consistently undershot analytical in prior
    # calibration (observed ratio ~0.66 on one design) -- search [0.3, 1.05]x
    a = 0.30 * y0_analytical
    b = 1.05 * y0_analytical

    print(f"\n=== {geo['name']}: y0_analytical={y0_analytical*1e3:.3f}mm, bracket [{a*1e3:.3f}, {b*1e3:.3f}]mm ===")

    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc = eval_s11(L, W, h, eps_r, c)
    fd = eval_s11(L, W, h, eps_r, d)

    for _ in range(N_ITERS):
        if fc < fd:  # c is better (more negative dB = deeper match)
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = eval_s11(L, W, h, eps_r, c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = eval_s11(L, W, h, eps_r, d)

    y0_true, s11_true = (c, fc) if fc < fd else (d, fd)
    print(f"-> BEST: y0_true={y0_true*1e3:.3f}mm, min_s11={s11_true:.2f}dB "
          f"(analytical was {y0_analytical*1e3:.3f}mm, ratio={y0_true/y0_analytical:.3f})")

    results.append({
        "name": geo["name"], "L": L, "W": W, "h": h, "eps_r": eps_r,
        "y0_analytical": float(y0_analytical), "y0_true": float(y0_true),
        "ratio": float(y0_true / y0_analytical), "min_s11_db": float(s11_true),
    })
    with open("data/m5_y0_bisect.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nAll geometries complete. Saved to data/m5_y0_bisect.json")