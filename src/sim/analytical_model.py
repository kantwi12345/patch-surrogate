import numpy as np
from scipy.special import sici, j0
from scipy.integrate import quad
from src.sim.schema import DesignParams, SimResult

C0 = 299_792_458.0
Z0_FEED = 50.0


def _eps_reff_and_deltaL(L, W, h, eps_r):
    eps_reff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** -0.5
    delta_L = 0.412 * h * (eps_reff + 0.3) * (W / h + 0.264) / (
        (eps_reff - 0.258) * (W / h + 0.8)
    )
    return eps_reff, delta_L


def _resonant_frequency(L, W, h, eps_r):
    eps_reff, delta_L = _eps_reff_and_deltaL(L, W, h, eps_r)
    Leff = L + 2 * delta_L
    fr = C0 / (2 * Leff * np.sqrt(eps_reff))
    return fr, eps_reff


def _slot_conductance_G1(W, k0):
    X0 = k0 * W
    Si_X0, _ = sici(X0)
    I1 = -2 + np.cos(X0) + X0 * Si_X0 + np.sin(X0) / X0
    return I1 / (120 * np.pi ** 2)


def _mutual_conductance_G12(W, L, k0):
    def integrand(theta):
        arg = 0.5 * k0 * W * np.cos(theta)
        f_term = (np.sin(arg) / np.cos(theta)) ** 2
        return f_term * j0(k0 * L * np.sin(theta)) * np.sin(theta) ** 3

    val, _ = quad(integrand, 1e-6, np.pi - 1e-6, limit=200)
    return val / (120 * np.pi ** 2)

def find_matched_inset(design: DesignParams, target_r: float = 50.0) -> float:
    """
    Solve analytically for the inset distance y0 that gives input
    resistance closest to target_r (typically 50 ohms), using the
    same two-slot model as simulate().
    """
    L, W, h, eps_r = design.L, design.W, design.h, design.eps_r
    fr, _ = _resonant_frequency(L, W, h, eps_r)
    lambda_r = C0 / fr
    k0_r = 2 * np.pi / lambda_r

    G1 = max(_slot_conductance_G1(W, k0_r), 1e-9)
    G12 = _mutual_conductance_G12(W, L, k0_r)
    Rin_edge = 1.0 / (2.0 * (G1 + G12))

    if Rin_edge <= target_r:
        # even the edge resistance is already <= target; best is y0=0
        return 0.0

    ratio = np.sqrt(target_r / Rin_edge)
    ratio = np.clip(ratio, -1.0, 1.0)
    y0 = (L / np.pi) * np.arccos(ratio)
    y0_max = 0.48 * L   # stay just inside the physical center
    return float(np.clip(y0, 0.0, y0_max))

def simulate(design: DesignParams, freq_hz: np.ndarray) -> SimResult:
    L, W, h, eps_r, y0 = design.L, design.W, design.h, design.eps_r, design.y0

    fr, eps_reff = _resonant_frequency(L, W, h, eps_r)
    lambda_r = C0 / fr
    k0_r = 2 * np.pi / lambda_r

    G1 = max(_slot_conductance_G1(W, k0_r), 1e-9)
    G12 = _mutual_conductance_G12(W, L, k0_r)

    Rin0 = 1.0 / (2.0 * (G1 + G12))
    Rin = max(Rin0 * np.cos(np.pi * y0 / L) ** 2, 1.0)

    bw_frac = 3.77 * ((eps_r - 1) / eps_r ** 2) * (h / lambda_r) * (W / L)
    bw_frac = float(np.clip(bw_frac, 0.003, 0.25))
    Q = 1.0 / bw_frac

    omega0 = 2 * np.pi * fr
    Cres = Q / (omega0 * Rin)
    Lres = 1.0 / (omega0 ** 2 * Cres)

    omega = 2 * np.pi * freq_hz
    Yin = 1.0 / Rin + 1j * omega * Cres + 1.0 / (1j * omega * Lres)
    Zin = 1.0 / Yin

    gamma = (Zin - Z0_FEED) / (Zin + Z0_FEED)
    s11_db = 20 * np.log10(np.clip(np.abs(gamma), 1e-6, None))
    # ---- gain model ----
    # NOTE: naive aperture formula (4*pi*Area/lambda^2) badly underestimates
    # patch directivity because patches are electrically small apertures;
    # real patch directivity is governed by the two-slot array pattern and
    # stays roughly in the 5-8 dBi range almost independent of absolute size.
    aspect = W / L
    D0_dbi = 6.5 + 1.5 * np.log10(np.clip(aspect, 0.5, 3.0))
    D0_dbi = float(np.clip(D0_dbi, 4.0, 9.0))

    eff = 1.0 - 0.03 * (eps_r - 1) - 12.0 * (h / lambda_r)
    eff = float(np.clip(eff, 0.3, 0.98))

    gain_dbi_res = D0_dbi + 10 * np.log10(eff)

    mismatch_lin = np.clip(1 - np.abs(gamma) ** 2, 1e-6, 1.0)
    gain_dbi = gain_dbi_res + 10 * np.log10(mismatch_lin)

    min_s11_db = float(np.min(s11_db))
    below_10 = s11_db <= -10.0
    if np.any(below_10):
        idx = np.where(below_10)[0]
        bandwidth_hz = float(freq_hz[idx[-1]] - freq_hz[idx[0]])
    else:
        bandwidth_hz = 0.0

    return SimResult(
        freq_hz=freq_hz,
        s11_db=s11_db,
        gain_dbi=gain_dbi,
        resonant_freq_hz=float(fr),
        bandwidth_hz=bandwidth_hz,
        min_s11_db=min_s11_db,
        peak_gain_dbi=float(np.max(gain_dbi)),
    )
def compute_resonance_params(design: DesignParams) -> dict:
    """
    Compute compact physical parameters DIRECTLY from geometry, using the
    exact same sub-formulas as simulate(). This guarantees reconstruct_curves()
    reproduces simulate()'s curves almost exactly -- no noisy re-extraction
    from an already-generated curve.
    """
    L, W, h, eps_r, y0 = design.L, design.W, design.h, design.eps_r, design.y0

    fr, eps_reff = _resonant_frequency(L, W, h, eps_r)
    lambda_r = C0 / fr
    k0_r = 2 * np.pi / lambda_r

    G1 = max(_slot_conductance_G1(W, k0_r), 1e-9)
    G12 = _mutual_conductance_G12(W, L, k0_r)
    Rin0 = 1.0 / (2.0 * (G1 + G12))
    Rin = max(Rin0 * np.cos(np.pi * y0 / L) ** 2, 1.0)

    bw_frac = 3.77 * ((eps_r - 1) / eps_r ** 2) * (h / lambda_r) * (W / L)
    bw_frac = float(np.clip(bw_frac, 0.003, 0.25))
    Q = 1.0 / bw_frac

    aspect = W / L
    D0_dbi = 6.5 + 1.5 * np.log10(np.clip(aspect, 0.5, 3.0))
    D0_dbi = float(np.clip(D0_dbi, 4.0, 9.0))
    eff = 1.0 - 0.03 * (eps_r - 1) - 12.0 * (h / lambda_r)
    eff = float(np.clip(eff, 0.3, 0.98))
    gain_premismatch = D0_dbi + 10 * np.log10(eff)

    # At exact resonance, parallel RLC reactance cancels -> Zin(fr) = Rin exactly
    gamma_at_res = (Rin - Z0_FEED) / (Rin + Z0_FEED)
    mismatch_at_res = max(1 - gamma_at_res ** 2, 1e-6)
    gain_at_res_dbi = gain_premismatch + 10 * np.log10(mismatch_at_res)

    return {
        "fr_hz": float(fr),
        "Q": float(np.clip(Q, 1.0, 500.0)),
        "Rin_matched": float(np.clip(Rin, 1.0, 2000.0)),
        "gain_at_res_dbi": float(gain_at_res_dbi),
    }


def reconstruct_curves(params: dict, freq_hz: np.ndarray) -> tuple:
    """
    Reconstruct S11(f) and gain(f) from compact params, using the exact
    same mismatch-coupling formula as simulate() (not a linear approximation).
    """
    fr = params["fr_hz"]
    Q = params["Q"]
    Rin = params["Rin_matched"]
    gain_at_res_dbi = params["gain_at_res_dbi"]

    omega0 = 2 * np.pi * fr
    Cres = Q / (omega0 * Rin)
    Lres = 1.0 / (omega0 ** 2 * Cres)

    omega = 2 * np.pi * freq_hz
    Yin = 1.0 / Rin + 1j * omega * Cres + 1.0 / (1j * omega * Lres)
    Zin = 1.0 / Yin

    gamma = (Zin - Z0_FEED) / (Zin + Z0_FEED)
    s11_db = 20 * np.log10(np.clip(np.abs(gamma), 1e-6, None))

    mismatch_lin = np.clip(1 - np.abs(gamma) ** 2, 1e-6, 1.0)
    gamma_at_res = (Rin - Z0_FEED) / (Rin + Z0_FEED)
    mismatch_at_res = max(1 - gamma_at_res ** 2, 1e-6)

    gain_dbi = gain_at_res_dbi + 10 * np.log10(mismatch_lin) - 10 * np.log10(mismatch_at_res)

    return s11_db, gain_dbi
