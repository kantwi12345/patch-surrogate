import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


PARAM_COLS = ["fr_hz", "Q", "Rin_matched", "gain_at_res_dbi", "gain_rolloff"]


class PatchParamDataset(Dataset):
    """
    Input (X): normalized geometry [L, W, h, eps_r, y0]
    Output (Y): normalized physical params [fr_hz, Q, Rin_matched, gain_at_res_dbi, gain_rolloff]
    Applies log-transform to fr_hz, Q, Rin_matched before z-scoring (all positive, skewed).
    """

    def __init__(self, df: pd.DataFrame, x_stats=None, y_stats=None):
        X = df[["L", "W", "h", "eps_r", "y0"]].to_numpy(dtype=np.float64)

        raw = df[PARAM_COLS].to_numpy(dtype=np.float64)
        # log-transform the positive, heavy-tailed columns: fr_hz, Q, Rin_matched
        transformed = raw.copy()
        transformed[:, 0] = np.log10(raw[:, 0])        # fr_hz
        transformed[:, 1] = np.log10(raw[:, 1])        # Q
        transformed[:, 2] = np.log10(raw[:, 2])        # Rin_matched
        # gain_at_res_dbi (idx 3) and gain_rolloff (idx 4) left as-is

        if x_stats is None:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0) + 1e-8
            x_stats = (x_mean, x_std)
        if y_stats is None:
            y_mean = transformed.mean(axis=0)
            y_std = transformed.std(axis=0) + 1e-8
            y_stats = (y_mean, y_std)

        self.x_mean, self.x_std = x_stats
        self.y_mean, self.y_std = y_stats

        self.X = ((X - self.x_mean) / self.x_std).astype(np.float32)
        self.Y = ((transformed - self.y_mean) / self.y_std).astype(np.float32)

        self.raw_params = raw  # keep original-scale values for eval/debugging

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

    def get_stats(self):
        return (self.x_mean, self.x_std), (self.y_mean, self.y_std)


def load_and_split_params(parquet_path="data/synthetic_dataset.parquet", val_frac=0.15, seed=42):
    df = pd.read_parquet(parquet_path)
    train_df, val_df = train_test_split(df, test_size=val_frac, random_state=seed)

    train_ds = PatchParamDataset(train_df)
    x_stats, y_stats = train_ds.get_stats()
    val_ds = PatchParamDataset(val_df, x_stats=x_stats, y_stats=y_stats)

    return train_ds, val_ds


def denormalize_params(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    """Inverse-transform normalized network output back into physical param dict."""
    raw = y_norm * y_std + y_mean
    fr_hz = 10 ** raw[..., 0]
    Q = 10 ** raw[..., 1]
    Rin = 10 ** raw[..., 2]
    gain_at_res = raw[..., 3]
    gain_rolloff = raw[..., 4]
    return {
        "fr_hz": fr_hz, "Q": Q, "Rin_matched": Rin,
        "gain_at_res_dbi": gain_at_res, "gain_rolloff": gain_rolloff,
    }