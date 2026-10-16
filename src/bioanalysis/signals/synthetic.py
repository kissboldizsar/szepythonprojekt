"""Szintetikus FID, spektrum és idősoros szenzorjelek."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalRecord:
    x: np.ndarray
    y: np.ndarray
    kind: str
    extra: dict


def generate_timeseries(n: int = 1024, seed: int | None = 0, noise: float = 0.35) -> SignalRecord:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, n)
    y = (
        1.2 * np.sin(2 * np.pi * 1.7 * t)
        + 0.6 * np.sin(2 * np.pi * 6.4 * t + 0.4)
        + 0.25 * np.sin(2 * np.pi * 14.0 * t)
        + rng.normal(0, noise, n)
    )
    return SignalRecord(x=t, y=y.astype(np.float64), kind="timeseries", extra={"fs": n / 8})


def generate_spectrum(n: int = 1024, seed: int | None = 0, noise: float = 0.08) -> SignalRecord:
    """FTIR-szerű abszorpciós spektrum Lorentz-csúcsokkal."""
    rng = np.random.default_rng(seed)
    wavenumber = np.linspace(400, 4000, n)
    y = 0.05 + 0.02 * np.sin(wavenumber / 180)
    peaks = [(720, 18, 0.7), (1100, 28, 0.9), (1650, 22, 0.55), (2920, 35, 0.8)]
    for center, width, height in peaks:
        y += height * (width**2) / ((wavenumber - center) ** 2 + width**2)
    y += rng.normal(0, noise, n)
    return SignalRecord(x=wavenumber, y=np.clip(y, 0, None), kind="spectrum", extra={})


def generate_fid(n: int = 1024, seed: int | None = 0, noise: float = 0.12) -> SignalRecord:
    """NMR-szerű szabad indukciós lecsengés komplex fázishibával."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.2, n)
    fid = (
        np.exp(-t / 0.28) * np.exp(1j * 2 * np.pi * 18 * t)
        + 0.55 * np.exp(-t / 0.18) * np.exp(1j * 2 * np.pi * 42 * t)
        + 0.3 * np.exp(-t / 0.12) * np.exp(1j * 2 * np.pi * 67 * t)
    )
    phi0 = 0.7
    phi1 = 1.8
    fid = fid * np.exp(1j * (phi0 + phi1 * t))
    fid = fid + rng.normal(0, noise, n) + 1j * rng.normal(0, noise, n)
    return SignalRecord(
        x=t,
        y=fid.astype(np.complex128),
        kind="fid",
        extra={"phi0": phi0, "phi1": phi1, "fs": n / 1.2},
    )
