"""0. és 1. rendű fázisjavítás FID / spektrum adatokhoz."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def apply_phase_correction(fid: np.ndarray, phi0: float, phi1: float = 0.0) -> np.ndarray:
    n = len(fid)
    ramp = np.linspace(-0.5, 0.5, n)
    return np.asarray(fid) * np.exp(-1j * (phi0 + phi1 * ramp))


def _entropy(real_spectrum: np.ndarray) -> float:
    mag = np.abs(real_spectrum)
    mag = mag / (mag.sum() + 1e-12)
    return float(-(mag * np.log(mag + 1e-12)).sum())


def estimate_phase(fid: np.ndarray) -> tuple[float, float]:
    """Fázisbecslés a valós spektrum negativitásának és entrópiájának minimalizálásával."""
    spectrum = np.fft.fftshift(np.fft.fft(fid))

    def objective(params: np.ndarray) -> float:
        phi0, phi1 = params
        ramp = np.linspace(-0.5, 0.5, len(spectrum))
        corrected = spectrum * np.exp(-1j * (phi0 + phi1 * ramp))
        real = corrected.real
        penalty = np.sum(np.clip(-real, 0, None) ** 2)
        return _entropy(real) + 0.15 * penalty

    result = minimize(objective, x0=np.array([0.0, 0.0]), method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])
