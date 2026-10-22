import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.data.dataset import load_and_split
from src.models.surrogate import PatchSurrogate


def train(
    epochs=200,
    batch_size=64,
    lr=1e-3,
    hidden_dim=256,
    n_hidden_layers=5,
    checkpoint_path="checkpoints/surrogate_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = load_and_split()
    n_freq = train_ds.n_freq_points
    output_dim = n_freq * 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PatchSurrogate(
        input_dim=5, output_dim=output_dim,
        hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            x_stats, y_stats = train_ds.get_stats()
            torch.save({
                "model_state": model.state_dict(),
                "x_mean": x_stats[0], "x_std": x_stats[1],
                "y_mean": y_stats[0], "y_std": y_stats[1],
                "n_freq_points": n_freq,
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            }, checkpoint_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | best {best_val_loss:.4f}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, history


if __name__ == "__main__":
    train()