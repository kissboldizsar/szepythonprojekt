from __future__ import annotations

import numpy as np


def intensity_grid(size: int = 80, seed: int | None = 1) -> np.ndarray:
    """Szintetikus 2D intenzitásmező spektrális / szöveti hőtérképhez."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    grid = (
        0.9 * np.exp(-((xx - 22) ** 2 + (yy - 28) ** 2) / 180)
        + 0.7 * np.exp(-((xx - 55) ** 2 + (yy - 50) ** 2) / 260)
        + 0.45 * np.exp(-((xx - 40) ** 2 + (yy - 18) ** 2) / 90)
        + 0.08 * rng.normal(size=(size, size))
    )
    return np.clip(grid, 0, None).astype(np.float32)


def threshold_mask(grid: np.ndarray, low: float, high: float) -> np.ndarray:
    return ((grid >= low) & (grid <= high)).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.2, 0.35)) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img[..., :3].astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    overlay = rgb.copy()
    binary = mask.astype(bool)
    for c in range(3):
        overlay[..., c] = np.where(binary, 0.45 * rgb[..., c] + 0.55 * color[c], rgb[..., c])
    return np.clip(overlay, 0, 1)
