"""
M4 step 7: small validation set spanning the trained frequency range,
run through both the analytical simulate() (ground truth per M3) and the
real HFSS simulate_hfss(), for direct comparison.
"""

import json
import numpy as np

from src.invert import design_from_spec
from src.sim.hfss_model import simulate_hfss

TARGET_FREQS_GHZ = [1.2, 1.8, 2.4, 3.0, 3.5, 4.2, 5.0, 5.8]

OUTPUT_PATH = "data/m4_validation_results.json"


def main():
    all_results = []

    for freq_ghz in TARGET_FREQS_GHZ:
        print(f"\n{'='*60}")
        print(f"Target frequency: {freq_ghz} GHz")
        print(f"{'='*60}")

        spec_result = design_from_spec(
            target_freq_hz=freq_ghz * 1e9,
            target_min_s11_db=-15.0,
        )
        design = spec_result["snapped_design"]
        analytical = spec_result["ground_truth"]

        print(f"\n--- Running HFSS for {freq_ghz} GHz design "
              f"(L={design.L*1e3:.2f}mm W={design.W*1e3:.2f}mm "
              f"h={design.h*1e3:.3f}mm eps_r={design.eps_r:.2f} y0={design.y0*1e3:.2f}mm) ---")

        freq_hz = np.linspace(freq_ghz * 1e9 * 0.8, freq_ghz * 1e9 * 1.2, 81)
        hfss_result = simulate_hfss(design, freq_hz, non_graphical=True)

        print(f"\n--- Comparison for {freq_ghz} GHz ---")
        print(f"Analytical: fr={analytical.resonant_freq_hz/1e9:.3f}GHz "
              f"min_s11={analytical.min_s11_db:.2f}dB peak_gain={analytical.peak_gain_dbi:.2f}dBi")
        print(f"HFSS:       fr={hfss_result.resonant_freq_hz/1e9:.3f}GHz "
              f"min_s11={hfss_result.min_s11_db:.2f}dB peak_gain={hfss_result.peak_gain_dbi:.2f}dBi")

        all_results.append({
            "target_freq_ghz": freq_ghz,
            "design": {
                "L": design.L, "W": design.W, "h": design.h,
                "eps_r": design.eps_r, "y0": design.y0,
            },
            "analytical": {
                "resonant_freq_hz": analytical.resonant_freq_hz,
                "min_s11_db": analytical.min_s11_db,
                "peak_gain_dbi": analytical.peak_gain_dbi,
                "bandwidth_hz": analytical.bandwidth_hz,
            },
            "hfss": {
                "resonant_freq_hz": hfss_result.resonant_freq_hz,
                "min_s11_db": hfss_result.min_s11_db,
                "peak_gain_dbi": hfss_result.peak_gain_dbi,
                "bandwidth_hz": hfss_result.bandwidth_hz,
            },
        })

        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved progress to {OUTPUT_PATH} ({len(all_results)}/{len(TARGET_FREQS_GHZ)} done)")

    print("\n\nAll designs complete.")


if __name__ == "__main__":
    main()