"""
M5 Step 1: sweep y0 at fixed (L,W,h,eps_r) in HFSS, extract Zin at the
design frequency only, to learn the TRUE Rin(y0) relationship and correct
the analytical find_matched_inset() formula.
"""
import json
import numpy as np

from src.sim.schema import DesignParams
from src.sim.hfss_model import build_geometry, setup_and_solve, extract_zin
from src.sim.analytical_model import _resonant_frequency

# Reuse the 8 validation geometries (L, W, h, eps_r), sweep y0 around
# the analytical-predicted value to bracket the true match point.
GEOMETRIES = [
    dict(name="1.2GHz_FR4",   L=0.05930, W=0.07600, h=0.00160,  eps_r=4.40),
    dict(name="1.8GHz_FR4",   L=0.03940, W=0.05070, h=0.00160,  eps_r=4.40),
    dict(name="2.4GHz_FR4",   L=0.02940, W=0.03800, h=0.00160,  eps_r=4.40),
    dict(name="3.0GHz_Alum",  L=0.01550, W=0.02110, h=0.00127,  eps_r=10.20),
    dict(name="3.5GHz_RO4003",L=0.02220, W=0.02840, h=0.001524, eps_r=3.55),
    dict(name="4.2GHz_RO3003",L=0.02000, W=0.02520, h=0.001524, eps_r=3.00),
    dict(name="5.0GHz_RO3003",L=0.01670, W=0.02120, h=0.001524, eps_r=3.00),
    dict(name="5.8GHz_RTD",   L=0.01700, W=0.02040, h=0.000787, eps_r=2.20),
]

N_Y0_POINTS = 6
results = []

for geo in GEOMETRIES:
    L, W, h, eps_r = geo["L"], geo["W"], geo["h"], geo["eps_r"]
    fr_est, _ = _resonant_frequency(L, W, h, eps_r)

    # Sweep y0 from near-edge to deep, covering the full plausible range
    y0_values = np.linspace(0.05 * L, 0.45 * L, N_Y0_POINTS)

    for y0 in y0_values:
        design = DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=float(y0))
        print(f"\n--- {geo['name']}: y0={y0*1e3:.2f}mm ({y0/L*100:.0f}% of L) ---")

        hfss, fr_est2 = build_geometry(
            design, non_graphical=True,
            create_port=True, create_airbox=True, use_inset=True,
        )
        try:
            setup_and_solve(
                hfss, fr_est2,
                freq_start_hz=fr_est2 * 0.85, freq_stop_hz=fr_est2 * 1.15,
                n_points=21,
            )
            freq_hz, re_zin, im_zin = extract_zin(hfss)
            idx = int(np.argmin(np.abs(im_zin)))
            Rin = float(re_zin[idx])
            Xin = float(im_zin[idx])
            res_freq = float(freq_hz[idx])
            print(f"Rin={Rin:.2f} ohm, Xin={Xin:.2f} ohm at {res_freq / 1e9:.3f} GHz (true Xin=0 crossing)")
        finally:
            hfss.release_desktop(close_projects=True, close_desktop=True)

        results.append({
            "name": geo["name"], "L": L, "W": W, "h": h, "eps_r": eps_r,
            "fr_est_hz": fr_est2, "y0": float(y0), "y0_frac": float(y0 / L),
            "Rin": Rin, "Xin": Xin, "resonant_freq_hz": res_freq,
        })

        with open("data/m5_y0_sweep.json", "w") as f:
            json.dump(results, f, indent=2)

print("\nAll y0 sweeps complete. Saved to data/m5_y0_sweep.json")