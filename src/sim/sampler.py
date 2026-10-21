import numpy as np
from scipy.stats import qmc
from src.sim.schema import DesignParams
from src.sim.analytical_model import find_matched_inset

C0 = 299_792_458.0

# Ranges are in NORMALIZED / ratio space -> this is what lets one model
# later cover multiple bands instead of overfitting to one frequency.
PARAM_RANGES = {
    "target_freq_ghz": (1.0, 6.0),
    "eps_r": (2.2, 10.2),
    "h_fraction": (0.01, 0.05),
    "L_jitter": (0.85, 1.15),
    "W_jitter": (0.85, 1.15),
    "inset_ratio": (0.0, 0.95),   # <-- was (0.0, 0.5); now y0 can approach L/2
}


def _eps_reff_and_deltaL(W, h, eps_r):
    eps_reff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** -0.5
    delta_L = 0.412 * h * (eps_reff + 0.3) * (W / h + 0.264) / (
        (eps_reff - 0.258) * (W / h + 0.8)
    )
    return eps_reff, delta_L


def nominal_design(target_freq_hz: float, eps_r: float, h: float):
    """Standard textbook patch design equations -> sane starting geometry."""
    W = C0 / (2 * target_freq_hz) * np.sqrt(2 / (eps_r + 1))
    eps_reff, delta_L = _eps_reff_and_deltaL(W, h, eps_r)
    L = C0 / (2 * target_freq_hz * np.sqrt(eps_reff)) - 2 * delta_L
    return L, W


def sample_designs(n_samples: int, seed: int = 42):
    keys = list(PARAM_RANGES.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    u = sampler.random(n_samples)
    lo = np.array([PARAM_RANGES[k][0] for k in keys])
    hi = np.array([PARAM_RANGES[k][1] for k in keys])
    samples = qmc.scale(u, lo, hi)

    designs = []
    for row in samples:
        vals = dict(zip(keys, row))
        f0_hz = vals["target_freq_ghz"] * 1e9
        eps_r = vals["eps_r"]
        lambda0 = C0 / f0_hz
        h = vals["h_fraction"] * lambda0

        L_nom, W_nom = nominal_design(f0_hz, eps_r, h)
        L = L_nom * vals["L_jitter"]
        W = W_nom * vals["W_jitter"]
        draft = DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=0.0)
        y0_match = find_matched_inset(draft, target_r=50.0)
        # inset_ratio now acts as a perturbation around the matched point (0.5 = exact match)
        y0 = y0_match * (0.8 + 0.42 * vals["inset_ratio"])  # range ~[0.5x, 1.45x] of matched inset
        y0 = np.clip(y0, 0.0, 0.48 * L)
        designs.append(DesignParams(L=L, W=W, h=h, eps_r=eps_r, y0=y0))
    return designs
