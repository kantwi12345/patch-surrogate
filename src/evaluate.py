import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import load_and_split
from src.models.surrogate import PatchSurrogate


def load_model(checkpoint_path="checkpoints/surrogate_best.pt"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = PatchSurrogate(
        input_dim=5,
        output_dim=ckpt["n_freq_points"] * 2,
        hidden_dim=ckpt["hidden_dim"],
        n_hidden_layers=ckpt["n_hidden_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def denormalize(arr, mean, std):
    return arr * std + mean


def evaluate(checkpoint_path="checkpoints/surrogate_best.pt", n_plot=6):
    model, ckpt = load_model(checkpoint_path)
    _, val_ds = load_and_split()

    n_freq = ckpt["n_freq_points"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    freq_grid = np.load("data/freq_grid_hz.npy") / 1e9  # GHz

    X = torch.from_numpy(val_ds.X).float()
    with torch.no_grad():
        preds_norm = model(X).numpy()

    preds = denormalize(preds_norm, y_mean, y_std)
    actual = denormalize(val_ds.Y, y_mean, y_std)

    pred_s11, pred_gain = preds[:, :n_freq], preds[:, n_freq:]
    actual_s11, actual_gain = actual[:, :n_freq], actual[:, n_freq:]

    # --- domain-relevant metrics ---
    pred_res_idx = np.argmin(pred_s11, axis=1)
    actual_res_idx = np.argmin(actual_s11, axis=1)
    pred_res_freq = freq_grid[pred_res_idx]
    actual_res_freq = freq_grid[actual_res_idx]
    res_freq_err_mhz = np.abs(pred_res_freq - actual_res_freq) * 1000

    pred_min_s11 = np.min(pred_s11, axis=1)
    actual_min_s11 = np.min(actual_s11, axis=1)
    min_s11_err_db = np.abs(pred_min_s11 - actual_min_s11)

    gain_err_db = np.abs(pred_gain - actual_gain).mean(axis=1)

    print("=== Domain-relevant validation metrics ===")
    print(f"Resonant freq error:  mean {res_freq_err_mhz.mean():.1f} MHz | median {np.median(res_freq_err_mhz):.1f} MHz | 95th pct {np.percentile(res_freq_err_mhz, 95):.1f} MHz")
    print(f"Min S11 error:        mean {min_s11_err_db.mean():.2f} dB | median {np.median(min_s11_err_db):.2f} dB | 95th pct {np.percentile(min_s11_err_db, 95):.2f} dB")
    print(f"Gain curve MAE:       mean {gain_err_db.mean():.2f} dB | median {np.median(gain_err_db):.2f} dB | 95th pct {np.percentile(gain_err_db, 95):.2f} dB")

    # --- plots ---
    idxs = np.random.choice(len(val_ds), n_plot, replace=False)
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 7))
    for i, idx in enumerate(idxs):
        axes[0, i].plot(freq_grid, actual_s11[idx], label="actual", color="black")
        axes[0, i].plot(freq_grid, pred_s11[idx], label="predicted", color="red", linestyle="--")
        axes[0, i].axhline(-10, color="gray", linestyle=":")
        axes[0, i].set_title(f"S11 sample {idx}")
        if i == 0:
            axes[0, i].legend()
            axes[0, i].set_ylabel("S11 (dB)")

        axes[1, i].plot(freq_grid, actual_gain[idx], label="actual", color="black")
        axes[1, i].plot(freq_grid, pred_gain[idx], label="predicted", color="red", linestyle="--")
        axes[1, i].set_title(f"Gain sample {idx}")
        if i == 0:
            axes[1, i].set_ylabel("Gain (dBi)")
        axes[1, i].set_xlabel("Freq (GHz)")

    plt.tight_layout()
    plt.savefig("checkpoints/validation_curves.png", dpi=120)
    print("Saved validation plot to checkpoints/validation_curves.png")
    plt.show()


if __name__ == "__main__":
    evaluate()