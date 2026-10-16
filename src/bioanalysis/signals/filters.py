from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, wiener


def denoise_signal(
    y: np.ndarray,
    method: str = "savgol",
    window: int = 21,
    polyorder: int = 3,
    cutoff: float = 0.08,
) -> np.ndarray:
    """Zajszűrés valós jeleken: Savitzky–Golay, Butterworth vagy Wiener."""
    signal = np.asarray(y)
    if np.iscomplexobj(signal):
        return denoise_signal(signal.real, method, window, polyorder, cutoff) + 1j * denoise_signal(
            signal.imag, method, window, polyorder, cutoff
        )

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

    method = method.lower()
    if method == "savgol":
        return savgol_filter(signal, window_length=min(window, len(signal) // 2 * 2 - 1 or 3), polyorder=polyorder)
    if method == "butterworth":
        b, a = butter(N=3, Wn=min(max(cutoff, 0.01), 0.49), btype="low")
        return filtfilt(b, a, signal)
    if method == "wiener":
        return wiener(signal, mysize=window)
    raise ValueError(f"Ismeretlen szűrő: {method}")
