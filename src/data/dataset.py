import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class PatchDataset(Dataset):
    """
    Wraps the synthetic (or later, real HFSS) dataset.
    Input (X): normalized geometry [L, W, h, eps_r, y0]
    Output (Y): concatenated [s11_db curve, gain_dbi curve]
    """

    def __init__(self, df: pd.DataFrame, x_stats=None, y_stats=None):
        # --- Build input matrix ---
        X = df[["L", "W", "h", "eps_r", "y0"]].to_numpy(dtype=np.float32)

        # --- Build output matrix (stack curves) ---
        s11 = np.stack(df["s11_db"].to_numpy())
        gain = np.stack(df["gain_dbi"].to_numpy())
        Y = np.concatenate([s11, gain], axis=1).astype(np.float32)

        # --- Normalization stats (compute from training set, reuse for val) ---
        if x_stats is None:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0) + 1e-8
            x_stats = (x_mean, x_std)
        if y_stats is None:
            y_mean = Y.mean(axis=0)
            y_std = Y.std(axis=0) + 1e-8
            y_stats = (y_mean, y_std)

        self.x_mean, self.x_std = x_stats
        self.y_mean, self.y_std = y_stats

        self.X_raw = X
        self.Y_raw = Y
        self.X = (X - self.x_mean) / self.x_std
        self.Y = (Y - self.y_mean) / self.y_std

        self.n_freq_points = s11.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
        )

    def get_stats(self):
        return (self.x_mean, self.x_std), (self.y_mean, self.y_std)


def load_and_split(parquet_path="data/synthetic_dataset.parquet", val_frac=0.15, seed=42):
    df = pd.read_parquet(parquet_path)
    train_df, val_df = train_test_split(df, test_size=val_frac, random_state=seed)

    train_ds = PatchDataset(train_df)
    x_stats, y_stats = train_ds.get_stats()
    val_ds = PatchDataset(val_df, x_stats=x_stats, y_stats=y_stats)

    return train_ds, val_ds