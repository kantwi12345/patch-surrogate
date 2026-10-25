"""
Diagnostic: run the FIXED simulate_hfss() on the 4.2 GHz design and confirm
it now correctly finds the true interior resonance near fr_est instead of
a sweep-edge artifact.
"""
import numpy as np
from src.sim.schema import DesignParams
from src.sim.hfss_model import simulate_hfss

design = DesignParams(
    L=0.02000, W=0.02520, h=0.001524, eps_r=3.00, y0=0.00710,
)

freq_hz = np.linspace(4.2e9 * 0.5, 4.2e9 * 1.5, 81)

result = simulate_hfss(design, freq_hz, non_graphical=True, n_points=81)

print("\n=== FIXED simulate_hfss() result ===")
print(f"resonant_freq_hz: {result.resonant_freq_hz/1e9:.3f} GHz")
print(f"min_s11_db:       {result.min_s11_db:.2f} dB")
print(f"bandwidth_hz:     {result.bandwidth_hz/1e6:.1f} MHz")
print(f"peak_gain_dbi:    {result.peak_gain_dbi:.2f} dBi")