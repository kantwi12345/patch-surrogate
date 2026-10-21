import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(path="data/synthetic_dataset.parquet"):
    return pd.read_parquet(path)


def plot_resonance_histogram(df):
    plt.figure(figsize=(6, 4))
    plt.hist(df["resonant_freq_hz"] / 1e9, bins=50)
    plt.xlabel("Resonant frequency (GHz)")
    plt.ylabel("Count")
    plt.title("Coverage of resonant frequencies across dataset")
    plt.show()


def plot_sample_curves(df, freq_grid_path="data/freq_grid_hz.npy", n=8):
    freq = np.load(freq_grid_path) / 1e9
    sample = df.sample(n, random_state=0)
    plt.figure(figsize=(7, 5))
    for _, row in sample.iterrows():
        plt.plot(freq, row["s11_db"], alpha=0.7)
    plt.axhline(-10, color="k", linestyle="--")
    plt.xlabel("Freq (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title(f"{n} random sampled S11 curves")
    plt.show()


def plot_length_vs_resonance(df):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["L"] * 1000, df["resonant_freq_hz"] / 1e9, s=4, alpha=0.4, c=df["eps_r"], cmap="viridis")
    plt.colorbar(label="eps_r")
    plt.xlabel("Patch length L (mm)")
    plt.ylabel("Resonant frequency (GHz)")
    plt.title("Sanity check: longer L / higher eps_r -> lower resonance")
    plt.show()


if __name__ == "__main__":
    df = load_dataset()
    print(df.describe())
    plot_resonance_histogram(df)
    plot_sample_curves(df)
    plot_length_vs_resonance(df)