import numpy as np
from src.sim.schema import DesignParams
from src.sim.hfss_model import simulate_hfss

design = DesignParams(
    L=0.02940, W=0.03800, h=0.0016, eps_r=4.40, y0=0.01090,
)

freq_hz = np.linspace(2.0e9, 2.8e9, 81)

result = simulate_hfss(design, freq_hz, non_graphical=False)

print("\n=== simulate_hfss() SimResult ===")
print(f"resonant_freq_hz: {result.resonant_freq_hz/1e9:.3f} GHz")
print(f"min_s11_db: {result.min_s11_db:.2f} dB")
print(f"bandwidth_hz: {result.bandwidth_hz/1e6:.1f} MHz")
print(f"peak_gain_dbi: {result.peak_gain_dbi:.2f} dBi")
print(f"s11_db array shape: {result.s11_db.shape}, first/last: {result.s11_db[0]:.2f} / {result.s11_db[-1]:.2f}")
print(f"gain_dbi array shape: {result.gain_dbi.shape}")