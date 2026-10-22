import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.sim.sampler import sample_designs
from src.sim.analytical_model import simulate, compute_resonance_params


FREQ_GRID_HZ = np.linspace(0.5e9, 8e9, 301)


def build(n_samples=5000, out_path="data/synthetic_dataset.parquet", seed=42):
    designs = sample_designs(n_samples, seed=seed)
    records = []
    n_skipped = 0

    for d in tqdm(designs, desc="Simulating (analytical placeholder)"):
        if d.L <= 0 or d.W <= 0 or d.h <= 0 or d.y0 >= d.L / 2:
            n_skipped += 1
            continue
        try:
            result = simulate(d, FREQ_GRID_HZ)
            params = compute_resonance_params(d)
        except Exception:
            n_skipped += 1
            continue

        records.append({
            "L": d.L, "W": d.W, "h": d.h, "eps_r": d.eps_r, "y0": d.y0,
            "resonant_freq_hz": result.resonant_freq_hz,
            "bandwidth_hz": result.bandwidth_hz,
            "min_s11_db": result.min_s11_db,
            "peak_gain_dbi": result.peak_gain_dbi,
            "s11_db": result.s11_db.tolist(),
            "gain_dbi": result.gain_dbi.tolist(),
            "fr_hz": params["fr_hz"],
            "Q": params["Q"],
            "Rin_matched": params["Rin_matched"],
            "gain_at_res_dbi": params["gain_at_res_dbi"],
            "gain_rolloff": params["gain_rolloff"],
        })

    df = pd.DataFrame(records)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    np.save("data/freq_grid_hz.npy", FREQ_GRID_HZ)

    print(f"Generated {len(df)} valid designs, skipped {n_skipped}")
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    build(n_samples=5000)