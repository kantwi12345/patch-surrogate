import numpy as np
import matplotlib.pyplot as plt
from src.sim.sampler import nominal_design
from src.sim.schema import DesignParams
from src.sim.analytical_model import find_matched_inset

target_f = 2.4e9
eps_r = 4.4         # FR4-like
h = 1.6e-3           # 1.6mm substrate, typical FR4
L, W = nominal_design(target_f, eps_r, h)
design = DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=0.3 * L / 2)

freq_grid = np.linspace(1.5e9, 3.5e9, 401)
result = simulate(design, freq_grid)

print(f"L={L*1000:.2f} mm, W={W*1000:.2f} mm")
print(f"Resonant freq: {result.resonant_freq_hz/1e9:.3f} GHz (target {target_f/1e9} GHz)")
print(f"Min S11: {result.min_s11_db:.2f} dB")
print(f"-10dB Bandwidth: {result.bandwidth_hz/1e6:.1f} MHz")
print(f"Peak realized gain: {result.peak_gain_dbi:.2f} dBi")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(freq_grid / 1e9, result.s11_db)
ax[0].axhline(-10, color="r", linestyle="--")
ax[0].set_xlabel("Freq (GHz)"); ax[0].set_ylabel("S11 (dB)"); ax[0].set_title("Return Loss")
ax[1].plot(freq_grid / 1e9, result.gain_dbi)
ax[1].set_xlabel("Freq (GHz)"); ax[1].set_ylabel("Realized Gain (dBi)"); ax[1].set_title("Gain")
plt.tight_layout()
plt.show()