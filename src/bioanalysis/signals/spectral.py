from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, spectrogram as sp_spectrogram


def power_spectrum(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(y)
    if np.iscomplexobj(signal):
        spec = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1 / fs))
        return freq, spec
    spec = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), d=1 / fs)
    return freq, spec


def spectrogram(y: np.ndarray, fs: float, nperseg: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(y)
    if np.iscomplexobj(signal):
        signal = signal.real
    nperseg = min(nperseg, max(16, len(signal) // 4))
    f, t, sxx = sp_spectrogram(signal, fs=fs, nperseg=nperseg)
    return f, t, sxx


def peak_table(x: np.ndarray, y: np.ndarray, prominence: float | None = None, max_peaks: int = 12) -> pd.DataFrame:
    y_abs = np.abs(np.asarray(y, dtype=np.float64))
    if prominence is None:
        prominence = 0.15 * (y_abs.max() - y_abs.min() + 1e-8)
    idx, props = find_peaks(y_abs, prominence=prominence)
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    idx = idx[order]
    return pd.DataFrame(
        {
            "x": np.asarray(x)[idx],
            "amplitude": y_abs[idx],
            "prominence": props["prominences"][order],
        }
    ).sort_values("x").reset_index(drop=True)
