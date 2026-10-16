import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioanalysis.segmentation.infer import segment_classical
from bioanalysis.segmentation.synthetic import generate_cell_image
from bioanalysis.segmentation.unet import UNet, bce_dice_loss
from bioanalysis.signals.filters import denoise_signal
from bioanalysis.signals.phase import apply_phase_correction, estimate_phase
from bioanalysis.signals.spectral import peak_table, power_spectrum
from bioanalysis.signals.synthetic import generate_fid, generate_spectrum, generate_timeseries
from bioanalysis.visualization.maps import intensity_grid, threshold_mask
import torch


def test_synthetic_cells_have_mask():
    sample = generate_cell_image(size=64, n_cells=6, seed=1, noise_std=0.02)
    assert sample.image.shape == (64, 64)
    assert sample.mask.max() == 1
    assert sample.n_cells == 6


def test_classical_segmentation_finds_cells():
    sample = generate_cell_image(size=64, n_cells=8, seed=2, noise_std=0.02)
    result = segment_classical(sample.image, min_size=8)
    assert result.n_objects >= 3


def test_unet_forward_and_loss():
    model = UNet()
    x = torch.rand(2, 1, 64, 64)
    y = torch.randint(0, 2, (2, 1, 64, 64)).float()
    logits = model(x)
    loss = bce_dice_loss(logits, y)
    assert logits.shape == y.shape
    assert torch.isfinite(loss)


def test_denoise_reduces_high_frequency_noise():
    rec = generate_timeseries(n=512, seed=0, noise=0.5)
    filtered = denoise_signal(rec.y, method="savgol", window=21)
    assert filtered.shape == rec.y.shape
    assert np.std(np.diff(filtered)) < np.std(np.diff(rec.y))


def test_spectrum_peaks():
    rec = generate_spectrum(n=800, seed=0, noise=0.02)
    peaks = peak_table(rec.x, rec.y, max_peaks=6)
    assert len(peaks) >= 3


def test_phase_correction_runs():
    rec = generate_fid(n=256, seed=1, noise=0.05)
    phi0, phi1 = estimate_phase(rec.y)
    corrected = apply_phase_correction(rec.y, phi0, phi1)
    freq, spec = power_spectrum(corrected, fs=float(rec.extra["fs"]))
    assert corrected.dtype == np.complex128
    assert len(freq) == len(spec)


def test_heatmap_threshold():
    grid = intensity_grid(size=40, seed=0)
    mask = threshold_mask(grid, 0.2, 0.9)
    assert mask.shape == grid.shape
    assert set(np.unique(mask)).issubset({0, 1})
