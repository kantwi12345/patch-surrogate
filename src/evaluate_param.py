import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.param_dataset import load_and_split_params, denormalize_params, PARAM_COLS
from src.models.surrogate import PatchSurrogate
from src.sim.analytical_model import reconstruct_curves


def load_model(checkpoint_path="checkpoints/surrogate_param_best.pt"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PatchSurrogate(
        input_dim=5, output_dim=4,
        hidden_dim=ckpt["hidden_dim"], n_hidden_layers=ckpt["n_hidden_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate(checkpoint_path="checkpoints/surrogate_param_best.pt",
             parquet_path="data/synthetic_dataset.parquet", n_plot=6):
    model, ckpt = load_model(checkpoint_path)
    _, val_ds = load_and_split_params()
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    freq_grid = np.load("data/freq_grid_hz.npy")
    freq_grid_ghz = freq_grid / 1e9

    X = torch.from_numpy(val_ds.X).float()
    with torch.no_grad():
        preds_norm = model(X).numpy()
    pred_params = denormalize_params(preds_norm, y_mean, y_std)

    # --- Pull the TRUE original simulate() curves for the val split ---
    df = pd.read_parquet(parquet_path)
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(df, test_size=0.15, random_state=42)
    true_s11 = np.stack(val_df["s11_db"].to_numpy())
    true_gain = np.stack(val_df["gain_dbi"].to_numpy())

    n = len(val_ds)
    pred_s11_curves = np.zeros((n, len(freq_grid)))
    pred_gain_curves = np.zeros((n, len(freq_grid)))
    for i in range(n):
        p_pred = {
            "fr_hz": pred_params["fr_hz"][i], "Q": pred_params["Q"][i],
            "Rin_matched": pred_params["Rin_matched"][i],
            "gain_at_res_dbi": pred_params["gain_at_res_dbi"][i],
        }
        pred_s11_curves[i], pred_gain_curves[i] = reconstruct_curves(p_pred, freq_grid)

    pred_min_s11 = np.min(pred_s11_curves, axis=1)
    true_min_s11 = np.min(true_s11, axis=1)
    min_s11_err_db = np.abs(pred_min_s11 - true_min_s11)

    pred_res_idx = np.argmin(pred_s11_curves, axis=1)
    true_res_idx = np.argmin(true_s11, axis=1)
    fr_err_mhz = np.abs(freq_grid[pred_res_idx] - freq_grid[true_res_idx]) / 1e6

    gain_curve_mae = np.abs(pred_gain_curves - true_gain).mean(axis=1)

    print("=== End-to-end domain metrics: predicted-params-reconstruction vs TRUE simulate() curves ===")
    print(f"Resonant freq error:  median {np.median(fr_err_mhz):.1f} MHz | 95th pct {np.percentile(fr_err_mhz,95):.1f} MHz")
    print(f"Min S11 error:        median {np.median(min_s11_err_db):.2f} dB | 95th pct {np.percentile(min_s11_err_db,95):.2f} dB")
    print(f"Gain curve MAE:       median {np.median(gain_curve_mae):.2f} dB | 95th pct {np.percentile(gain_curve_mae,95):.2f} dB")

    # --- Sanity check: reconstruction fidelity using GROUND-TRUTH params (no network) ---
    true_raw = val_ds.raw_params
    recon_s11_from_truth = np.zeros((n, len(freq_grid)))
    recon_gain_from_truth = np.zeros((n, len(freq_grid)))
    for i in range(n):
        p_true = {
            "fr_hz": true_raw[i, 0], "Q": true_raw[i, 1],
            "Rin_matched": true_raw[i, 2], "gain_at_res_dbi": true_raw[i, 3],
        }
        recon_s11_from_truth[i], recon_gain_from_truth[i] = reconstruct_curves(p_true, freq_grid)

    round_trip_s11_err = np.abs(np.min(recon_s11_from_truth, axis=1) - true_min_s11)
    round_trip_gain_err = np.abs(recon_gain_from_truth - true_gain).mean(axis=1)
    print("\n=== Round-trip reconstruction fidelity (ground-truth params, NO network involved) ===")
    print(f"Min S11 round-trip error:  median {np.median(round_trip_s11_err):.3f} dB (should be ~0)")
    print(f"Gain round-trip MAE:       median {np.median(round_trip_gain_err):.3f} dB (should be ~0)")

    # --- plots ---
    idxs = np.random.choice(n, n_plot, replace=False)
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 7))
    for i, idx in enumerate(idxs):
        axes[0, i].plot(freq_grid_ghz, true_s11[idx], label="true sim", color="black")
        axes[0, i].plot(freq_grid_ghz, pred_s11_curves[idx], label="predicted", color="red", linestyle="--")
        axes[0, i].axhline(-10, color="gray", linestyle=":")
        axes[0, i].set_title(f"S11 sample {idx}")
        if i == 0:
            axes[0, i].legend()
            axes[0, i].set_ylabel("S11 (dB)")

        axes[1, i].plot(freq_grid_ghz, true_gain[idx], label="true sim", color="black")
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