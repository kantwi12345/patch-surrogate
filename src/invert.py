"""
M3 — Inverse design: target spec -> candidate geometry.

Given a target (resonant frequency, minimum return loss, optional gain),
runs gradient descent directly on the geometry input of the frozen M2.5
parametric surrogate (not the dense-curve model), using a sigmoid
reparameterization to keep geometry inside the physically sane region the
surrogate was trained on. Multi-start to avoid local minima. Best result is
snapped to a manufacturable substrate choice, then validated against the
real analytical simulate() (ground truth), not the surrogate.
"""

import argparse

import numpy as np
import torch

from src.sim.schema import DesignParams
from src.sim.sampler import nominal_design
from src.sim.analytical_model import simulate, find_matched_inset
from src.models.surrogate import PatchSurrogate

CHECKPOINT_PATH = "checkpoints/surrogate_param_best.pt"
FREQ_GRID_PATH = "data/freq_grid_hz.npy"

# Physical search-box bounds for the optimizer, loosely derived from the
# sampler's achievable ranges (target_freq 1-6GHz, eps_r 2.2-10.2,
# h_fraction 1-5% of lambda0, L/W jitter 0.85-1.15x nominal). Padded a bit
# on each side. y0 is NOT given a fixed box -- it's parameterized as a
# fraction of 0.48*L so it's always physically valid regardless of L.
GEOM_BOUNDS = {
    "L":     (0.004, 0.15),
    "W":     (0.008, 0.14),
    "h":     (0.0004, 0.016),
    "eps_r": (2.0, 10.5),
}

# Small catalog of common substrate (eps_r, h) choices for the final
# "snap to manufacturable" step.
STANDARD_SUBSTRATES = [
    (2.2, 0.787e-3,  "RT/duroid 5880, 31mil"),
    (2.2, 1.575e-3,  "RT/duroid 5880, 62mil"),
    (3.0, 1.524e-3,  "RO3003, 60mil"),
    (3.55, 0.508e-3, "RO4003C, 20mil"),
    (3.55, 1.524e-3, "RO4003C, 60mil"),
    (4.4, 0.8e-3,    "FR4, 0.8mm"),
    (4.4, 1.6e-3,    "FR4, 1.6mm"),
    (6.15, 1.27e-3,  "RO3006, 50mil"),
    (10.2, 0.635e-3, "RO3010 / Alumina, 25mil"),
    (10.2, 1.27e-3,  "RO3010 / Alumina, 50mil"),
]


def load_surrogate(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PatchSurrogate(
        input_dim=5, output_dim=4,
        hidden_dim=ckpt["hidden_dim"], n_hidden_layers=ckpt["n_hidden_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stats = {
        "x_mean": torch.tensor(np.asarray(ckpt["x_mean"]), dtype=torch.float32, device=device),
        "x_std":  torch.tensor(np.asarray(ckpt["x_std"]),  dtype=torch.float32, device=device),
        "y_mean": torch.tensor(np.asarray(ckpt["y_mean"]), dtype=torch.float32, device=device),
        "y_std":  torch.tensor(np.asarray(ckpt["y_std"]),  dtype=torch.float32, device=device),
    }
    return model, stats


def denormalize_params_torch(y_norm, y_mean, y_std):
    """Differentiable torch equivalent of param_dataset.denormalize_params()."""
    raw = y_norm * y_std + y_mean
    fr_hz = torch.pow(10.0, raw[..., 0])
    Q = torch.pow(10.0, raw[..., 1])
    Rin = torch.pow(10.0, raw[..., 2])
    gain_at_res = raw[..., 3]
    return fr_hz, Q, Rin, gain_at_res


def unconstrained_to_geometry(theta, bounds):
    """
    theta: [B,5] raw unconstrained optimizer params -> geometry [B,5] =
    [L, W, h, eps_r, y0], via sigmoid reparam into physical bounds. y0 is
    derived as a fraction of 0.48*L so it's always valid (0 <= y0 < L/2)
    regardless of the other params -- this is also exactly the quantity
    we're searching over to find the 50 ohm match point.
    """
    sig = torch.sigmoid(theta)

    def scale(frac, lo, hi):
        return lo + frac * (hi - lo)

    L = scale(sig[:, 0], *bounds["L"])
    W = scale(sig[:, 1], *bounds["W"])
    h = scale(sig[:, 2], *bounds["h"])
    eps_r = scale(sig[:, 3], *bounds["eps_r"])
    y0_frac = sig[:, 4]
    y0 = y0_frac * 0.48 * L

    return torch.stack([L, W, h, eps_r, y0], dim=1)


def predict_params(model, stats, geom):
    x_norm = (geom - stats["x_mean"]) / stats["x_std"]
    y_norm = model(x_norm)
    return denormalize_params_torch(y_norm, stats["y_mean"], stats["y_std"])


def spec_loss(
    fr_hz, Rin, gain_at_res_dbi,
    target_freq_hz, target_min_s11_db, target_gain_dbi=None,
    w_freq=1.0, w_s11=1.0, w_gain=0.3, w_gain_reg=0.02,
):
    """
    Loss computed directly on the compact predicted params (closed-form,
    differentiable) rather than the full reconstructed curve -- equivalent
    for a freq/min-S11/gain spec, since min-S11-at-resonance is exactly
    (Rin-50)/(Rin+50).
    """
    freq_loss = ((fr_hz - target_freq_hz) / target_freq_hz) ** 2

    gamma_at_res = torch.clamp(torch.abs((Rin - 50.0) / (Rin + 50.0)), min=1e-6)
    pred_min_s11_db = 20 * torch.log10(gamma_at_res)
    # only penalize if NOT deep enough; deeper-than-target match is free
    s11_loss = torch.clamp(pred_min_s11_db - target_min_s11_db, min=0.0) ** 2

    if target_gain_dbi is not None:
        gain_loss = (gain_at_res_dbi - target_gain_dbi) ** 2
    else:
        # mild regularizer nudging toward higher gain, not a hard target
        gain_loss = torch.clamp(5.0 - gain_at_res_dbi, min=0.0) * w_gain_reg

    total = w_freq * freq_loss + w_s11 * s11_loss + w_gain * gain_loss
    return total, pred_min_s11_db


def run_inversion(
    target_freq_hz,
    target_min_s11_db=-15.0,
    target_gain_dbi=None,
    n_starts=40,
    n_steps=400,
    lr=0.05,
    seed=0,
    device="cpu",
):
    torch.manual_seed(seed)
    model, stats = load_surrogate(device=device)

    theta = torch.empty(n_starts, 5, device=device).uniform_(-3.0, 3.0)
    theta.requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    target_freq_t = torch.tensor(float(target_freq_hz), device=device)
    target_gain_t = None if target_gain_dbi is None else torch.tensor(float(target_gain_dbi), device=device)

    for step in range(n_steps):
        optimizer.zero_grad()
        geom = unconstrained_to_geometry(theta, GEOM_BOUNDS)
        fr_hz, Q, Rin, gain_at_res = predict_params(model, stats, geom)
        total, pred_s11 = spec_loss(
            fr_hz, Rin, gain_at_res, target_freq_t, target_min_s11_db, target_gain_t,
        )
        loss = total.mean()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            best_idx = total.argmin().item()
            print(f"step {step:4d} | mean_loss {loss.item():.5f} | best_loss {total[best_idx].item():.5f} "
                  f"| best fr {fr_hz[best_idx].item()/1e9:.3f} GHz | best S11 {pred_s11[best_idx].item():.2f} dB")

    with torch.no_grad():
        geom = unconstrained_to_geometry(theta, GEOM_BOUNDS)
        fr_hz, Q, Rin, gain_at_res = predict_params(model, stats, geom)
        total, pred_s11 = spec_loss(
            fr_hz, Rin, gain_at_res, target_freq_t, target_min_s11_db, target_gain_t,
        )

    order = torch.argsort(total)
    geom_sorted = geom[order].cpu().numpy()
    total_sorted = total[order].cpu().numpy()
    fr_sorted = fr_hz[order].cpu().numpy()
    s11_sorted = pred_s11[order].cpu().numpy()
    gain_sorted = gain_at_res[order].cpu().numpy()

    results = []
    for i in range(len(order)):
        results.append({
            "L": float(geom_sorted[i, 0]), "W": float(geom_sorted[i, 1]),
            "h": float(geom_sorted[i, 2]), "eps_r": float(geom_sorted[i, 3]),
            "y0": float(geom_sorted[i, 4]),
            "loss": float(total_sorted[i]),
            "pred_fr_hz": float(fr_sorted[i]),
            "pred_min_s11_db": float(s11_sorted[i]),
            "pred_gain_at_res_dbi": float(gain_sorted[i]),
        })
    return results


def snap_to_manufacturable(design_dict, target_freq_hz):
    """
    Snap eps_r/h to the nearest standard substrate from a small catalog,
    then regenerate L, W via the closed-form nominal_design() at the target
    frequency, and re-solve y0 via find_matched_inset() for a fresh 50 ohm
    match on the snapped substrate. This is more robust than literally
    rounding the raw optimized L/W/y0 floats, since eps_r/h are the only
    truly discretized choices in real fabrication; L/W/y0 get cut to
    whatever precision is needed (rounded here to 0.1mm).
    """
    eps_r_opt, h_opt = design_dict["eps_r"], design_dict["h"]

    best = min(
        STANDARD_SUBSTRATES,
        key=lambda s: ((np.log(s[0]) - np.log(eps_r_opt)) ** 2
                       + (np.log(s[1]) - np.log(h_opt)) ** 2),
    )
    eps_r_snap, h_snap, name = best

    L_nom, W_nom = nominal_design(target_freq_hz, eps_r_snap, h_snap)
    draft = DesignParams(L=L_nom, W=W_nom, h=h_snap, eps_r=eps_r_snap, y0=0.0)
    y0_snap = find_matched_inset(draft, target_r=50.0)

    L_final = round(float(L_nom), 4)
    W_final = round(float(W_nom), 4)
    y0_final = round(float(y0_snap), 4)

    return DesignParams(L=L_final, W=W_final, h=h_snap, eps_r=eps_r_snap, y0=y0_final), name


def design_from_spec(
    target_freq_hz,
    target_min_s11_db=-15.0,
    target_gain_dbi=None,
    n_starts=40,
    n_steps=400,
    lr=0.05,
    top_k=5,
    seed=0,
):
    freq_grid = np.load(FREQ_GRID_PATH)

    results = run_inversion(
        target_freq_hz=target_freq_hz,
        target_min_s11_db=target_min_s11_db,
        target_gain_dbi=target_gain_dbi,
        n_starts=n_starts, n_steps=n_steps, lr=lr, seed=seed,
    )

    print("\n=== Top candidates (surrogate-predicted) ===")
    for i, r in enumerate(results[:top_k]):
        print(f"[{i}] loss={r['loss']:.5f}  L={r['L']*1e3:.2f}mm W={r['W']*1e3:.2f}mm "
              f"h={r['h']*1e3:.3f}mm eps_r={r['eps_r']:.2f} y0={r['y0']*1e3:.2f}mm | "
              f"pred_fr={r['pred_fr_hz']/1e9:.3f}GHz pred_S11={r['pred_min_s11_db']:.2f}dB "
              f"pred_gain={r['pred_gain_at_res_dbi']:.2f}dBi")

    best = results[0]
    snapped_design, substrate_name = snap_to_manufacturable(best, target_freq_hz)

    print(f"\n=== Snapped to manufacturable substrate: {substrate_name} ===")
    print(f"L={snapped_design.L*1e3:.2f}mm  W={snapped_design.W*1e3:.2f}mm  "
          f"h={snapped_design.h*1e3:.3f}mm  eps_r={snapped_design.eps_r:.2f}  "
          f"y0={snapped_design.y0*1e3:.2f}mm")

    result = simulate(snapped_design, freq_grid)
    print("\n=== Ground-truth validation (simulate(), NOT the surrogate) ===")
    print(f"True resonant freq: {result.resonant_freq_hz/1e9:.3f} GHz  (target {target_freq_hz/1e9:.3f} GHz)")
    print(f"True min S11:       {result.min_s11_db:.2f} dB  (target <= {target_min_s11_db:.1f} dB)")
    print(f"True peak gain:     {result.peak_gain_dbi:.2f} dBi")
    print(f"True bandwidth:     {result.bandwidth_hz/1e6:.1f} MHz")

    return {
        "candidates": results[:top_k],
        "snapped_design": snapped_design,
        "substrate_name": substrate_name,
        "ground_truth": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3: spec -> geometry inverse design")
    parser.add_argument("--freq_ghz", type=float, required=True, help="target resonant frequency, GHz")
    parser.add_argument("--min_s11_db", type=float, default=-15.0, help="target minimum return loss, dB (negative)")
    parser.add_argument("--gain_dbi", type=float, default=None, help="optional target gain, dBi")
    parser.add_argument("--n_starts", type=int, default=40)
    parser.add_argument("--n_steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    design_from_spec(
        target_freq_hz=args.freq_ghz * 1e9,
        target_min_s11_db=args.min_s11_db,
        target_gain_dbi=args.gain_dbi,
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        lr=args.lr,
        seed=args.seed,
    )