from .filters import denoise_signal
from .phase import apply_phase_correction, estimate_phase
from .spectral import peak_table, power_spectrum, spectrogram
from .synthetic import generate_fid, generate_spectrum, generate_timeseries

__all__ = [
    "denoise_signal",
    "apply_phase_correction",
    "estimate_phase",
    "power_spectrum",
    "spectrogram",
    "peak_table",
    "generate_fid",
    "generate_spectrum",
    "generate_timeseries",
]
