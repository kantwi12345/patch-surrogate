from dataclasses import dataclass
import numpy as np


@dataclass
class DesignParams:
    L: float          # patch length, meters
    W: float          # patch width, meters
    h: float          # substrate height, meters
    eps_r: float      # relative permittivity
    y0: float         # feed inset distance from patch edge, meters
    loss_tangent: float = 0.001

    def as_array(self) -> np.ndarray:
        return np.array([self.L, self.W, self.h, self.eps_r, self.y0], dtype=np.float64)


@dataclass
class SimResult:
    freq_hz: np.ndarray
    s11_db: np.ndarray
    gain_dbi: np.ndarray
    resonant_freq_hz: float
    bandwidth_hz: float
    min_s11_db: float
    peak_gain_dbi: float