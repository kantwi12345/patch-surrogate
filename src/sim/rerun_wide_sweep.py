"""
Re-run the 4 designs whose HFSS resonance was clipped at the sweep boundary,
using a much wider sweep (±50% around target) to capture the true dip.
"""

import json
import numpy as np

from src.sim.schema import DesignParams
from src.sim.hfss_model import simulate_hfss

TARGET_FREQS_GHZ = [3.5, 4.2, 5.0, 5.8]  # only the 4 clipped designs
INPUT_JSON = "data/m4_validation_results.json"
OUTPUT_JSON = "data/m4_validation_results_wide.json"

def main():
    with open(INPUT_JSON, "r") as f:
        all_results = json.load(f)

    for entry in all_results:
        if entry["target_freq_ghz"] not in TARGET_FREQS_GHZ:
            continue

        print(f"\n{'='*60}")
        print(f"Re-running {entry['target_freq_ghz']} GHz with WIDE sweep")
        print(f"{'='*60}")

        design = DesignParams(**entry["design"])
        freq_hz = np.linspace(
            entry["target_freq_ghz"] * 1e9 * 0.5,  # 50% below
            entry["target_freq_ghz"] * 1e9 * 1.5,  # 50% above
            81,
        )

        hfss_result = simulate_hfss(design, freq_hz, non_graphical=True)

        print(f"\n--- Wide-sweep comparison for {entry['target_freq_ghz']} GHz ---")
        print(f"Analytical: fr={entry['analytical']['resonant_freq_hz']/1e9:.3f}GHz "
              f"min_s11={entry['analytical']['min_s11_db']:.2f}dB "
              f"peak_gain={entry['analytical']['peak_gain_dbi']:.2f}dBi")
        print(f"HFSS (wide): fr={hfss_result.resonant_freq_hz/1e9:.3f}GHz "
              f"min_s11={hfss_result.min_s11_db:.2f}dB "
              f"peak_gain={hfss_result.peak_gain_dbi:.2f}dBi")

        # Overwrite the HFSS result in the entry
        entry["hfss"] = {
            "resonant_freq_hz": hfss_result.resonant_freq_hz,
            "min_s11_db": hfss_result.min_s11_db,
            "peak_gain_dbi": hfss_result.peak_gain_dbi,
            "bandwidth_hz": hfss_result.bandwidth_hz,
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved progress to {OUTPUT_JSON}")

    print("\nAll wide-sweep re-runs complete.")

if __name__ == "__main__":
    main()