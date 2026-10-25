"""
Isolated single-design test to check whether the 4.2 GHz 'runaway resonance'
result is a real HFSS effect or a material-caching bug from running multiple
designs sequentially in one process.
"""

import numpy as np
from src.sim.schema import DesignParams
from src.sim.hfss_model import simulate_hfss

design = DesignParams(
    L=0.02000, W=0.02520, h=0.001524, eps_r=3.00, y0=0.00710,
)

freq_hz = np.linspace(2.1e9, 6.3e9, 81)  # same ±50% wide window as before

result = simulate_hfss(design, freq_hz, non_graphical=False)

print("\n=== Isolated 4.2 GHz test ===")
print(f"resonant_freq_hz: {result.resonant_freq_hz/1e9:.3f} GHz")
print(f"min_s11_db: {result.min_s11_db:.2f} dB")
print(f"peak_gain_dbi: {result.peak_gain_dbi:.2f} dBi")
