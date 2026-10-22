import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.param_dataset import load_and_split_params, denormalize_params, PARAM_COLS
from src.models.surrogate import PatchSurrogate
from src.sim.analytical_model import reconstruct_curves


def load_model(checkpoint_path="checkpoints/surrogate_param_best.pt"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PatchSurrogate(
        input_dim=5, output_dim=5,
        hidden_dim=ckpt["hidden_dim"], n_hidden_layers=ckpt["n_hidden_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate(checkpoint_path="checkpoints/surrogate_param_best.pt", n_plot=6):
    model, ckpt = load_model(checkpoint_path)
    _, val_ds = load_and_split_params()
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    freq_grid = np.load("data/freq_grid_hz.npy")
    freq_grid_ghz = freq_grid / 1e9

    X = torch.from_numpy(val_ds.X).float()
    with torch.no_grad():
        preds_norm = model(X).numpy()

    pred_params = denormalize_params(preds_norm, y_mean, y_std)
    actual_raw = val_ds.raw_params  # shape (N, 5), columns = PARAM_COLS order

    # --- direct parameter errors ---
    fr_err_mhz = np.abs(pred_params["fr_hz"] - actual_raw[:, 0]) / 1e6
    Q_err_pct = np.abs(pred_params["Q"] - actual_raw[:, 1]) / actual_raw[:, 1] * 100
    Rin_err_pct = np.abs(pred_params["Rin_matched"] - actual_raw[:, 2]) / actual_raw[:, 2] * 100
    gain_err_db = np.abs(pred_params["gain_at_res_dbi"] - actual_raw[:, 3])

    print("=== Parameter-level errors ===")
    print(f"Resonant freq error:  median {np.median(fr_err_mhz):.1f} MHz | 95th pct {np.percentile(fr_err_mhz,95):.1f} MHz")
    print(f"Q error:              median {np.median(Q_err_pct):.1f}% | 95th pct {np.percentile(Q_err_pct,95):.1f}%")
    print(f"Rin error:            median {np.median(Rin_err_pct):.1f}% | 95th pct {np.percentile(Rin_err_pct,95):.1f}%")
    print(f"Gain-at-res error:    median {np.median(gain_err_db):.2f} dB | 95th pct {np.percentile(gain_err_db,95):.2f} dB")

    # --- reconstruct full curves for both predicted and actual params, compare ---
    n = len(val_ds)
    pred_s11_curves = np.zeros((n, len(freq_grid)))
    actual_s11_curves = np.zeros((n, len(freq_grid)))
    pred_gain_curves = np.zeros((n, len(freq_grid)))
    actual_gain_curves = np.zeros((n, len(freq_grid)))

    for i in range(n):
        p_pred = {
            "fr_hz": pred_params["fr_hz"][i], "Q": pred_params["Q"][i],
            "Rin_matched": pred_params["Rin_matched"][i],
            "gain_at_res_dbi": pred_params["gain_at_res_dbi"][i],
            "gain_rolloff": pred_params["gain_rolloff"][i],
        }
        p_actual = {
            "fr_hz": actual_raw[i, 0], "Q": actual_raw[i, 1],
            "Rin_matched": actual_raw[i, 2], "gain_at_res_dbi": actual_raw[i, 3],
            "gain_rolloff": actual_raw[i, 4],
        }
        pred_s11_curves[i], pred_gain_curves[i] = reconstruct_curves(p_pred, freq_grid)
        actual_s11_curves[i], actual_gain_curves[i] = reconstruct_curves(p_actual, freq_grid)

    pred_min_s11 = np.min(pred_s11_curves, axis=1)
    actual_min_s11 = np.min(actual_s11_curves, axis=1)
    min_s11_err_db = np.abs(pred_min_s11 - actual_min_s11)

    pred_res_idx = np.argmin(pred_s11_curves, axis=1)
    actual_res_idx = np.argmin(actual_s11_curves, axis=1)
    recon_fr_err_mhz = np.abs(freq_grid[pred_res_idx] - freq_grid[actual_res_idx]) / 1e6

    gain_curve_mae = np.abs(pred_gain_curves - actual_gain_curves).mean(axis=1)

    print("\n=== Reconstructed-curve domain metrics (comparable to M2 dense-curve run) ===")
    print(f"Resonant freq error:  median {np.median(recon_fr_err_mhz):.1f} MHz | 95th pct {np.percentile(recon_fr_err_mhz,95):.1f} MHz")
    print(f"Min S11 error:        median {np.median(min_s11_err_db):.2f} dB | 95th pct {np.percentile(min_s11_err_db,95):.2f} dB")
    print(f"Gain curve MAE:       median {np.median(gain_curve_mae):.2f} dB | 95th pct {np.percentile(gain_curve_mae,95):.2f} dB")

    # --- plots ---
    idxs = np.random.choice(n, n_plot, replace=False)
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 7))
    for i, idx in enumerate(idxs):
        axes[0, i].plot(freq_grid_ghz, actual_s11_curves[idx], label="actual", color="black")
        axes[0, i].plot(freq_grid_ghz, pred_s11_curves[idx], label="predicted", color="red", linestyle="--")
        axes[0, i].axhline(-10, color="gray", linestyle=":")
        axes[0, i].set_title(f"S11 sample {idx}")
        if i == 0:
            axes[0, i].legend()
            axes[0, i].set_ylabel("S11 (dB)")

        axes[1, i].plot(freq_grid_ghz, actual_gain_curves[idx], label="actual", color="black")
        axes[1, i].plot(freq_grid_ghz, pred_gain_curves[idx], label="predicted", color="red", linestyle="--")
        axes[1, i].set_title(f"Gain sample {idx}")
        if i == 0:
            axes[1, i].set_ylabel("Gain (dBi)")
        axes[1, i].set_xlabel("Freq (GHz)")

    plt.tight_layout()
    plt.savefig("checkpoints/validation_curves_param.png", dpi=120)
    print("\nSaved validation plot to checkpoints/validation_curves_param.png")
    plt.show()


if __name__ == "__main__":
    evaluate()