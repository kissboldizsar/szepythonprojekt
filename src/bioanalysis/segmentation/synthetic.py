"""Szintetikus mikroszkópos felvételek: fluoreszcens sejtek sötét háttéren."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticImage:
    image: np.ndarray
    mask: np.ndarray
    centers: np.ndarray
    n_cells: int


def generate_cell_image(
    size: int = 128,
    n_cells: int | None = None,
    seed: int | None = None,
    noise_std: float = 0.08,
) -> SyntheticImage:
    """Generál egy fluoreszcens mikroszkópszerű képet és bináris maszkot."""
    rng = np.random.default_rng(seed)
    if n_cells is None:
        n_cells = int(rng.integers(8, 22))

    image = rng.normal(0.08, 0.02, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    # Egyenetlen megvilágítás
    image += 0.12 * np.exp(-((xx - size * 0.4) ** 2 + (yy - size * 0.3) ** 2) / (2 * (size * 0.45) ** 2))

    mask = np.zeros((size, size), dtype=np.uint8)
    centers = []

    for _ in range(n_cells):
        cy = float(rng.uniform(8, size - 8))
        cx = float(rng.uniform(8, size - 8))
        rx = float(rng.uniform(3.5, 8.5))
        ry = float(rng.uniform(3.5, 8.5))
        intensity = float(rng.uniform(0.55, 1.0))
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        image += intensity * blob
        mask = np.maximum(mask, (blob > 0.35).astype(np.uint8))
        centers.append((cy, cx))

    image = gaussian_filter(image, sigma=0.6)
    image += rng.normal(0, noise_std, image.shape).astype(np.float32)
    image = np.clip(image, 0, None)
    image /= image.max() + 1e-8

    return SyntheticImage(
        image=image.astype(np.float32),
        mask=mask,
        centers=np.asarray(centers, dtype=np.float32),
        n_cells=n_cells,
    )
