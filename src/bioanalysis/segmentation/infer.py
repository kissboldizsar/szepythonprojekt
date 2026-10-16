"""Szegmentálás U-Net-tel, klasszikus küszöböléses tartalékkal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from .train import DEFAULT_WEIGHTS
from .unet import UNet


@dataclass
class SegmentResult:
    probability: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    n_objects: int
    method: str


def _to_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def load_model(weights_path: Path | None = None, device: str | None = None) -> tuple[UNet, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, device
    raise FileNotFoundError(str(path))


@torch.no_grad()
def segment_with_unet(
    image: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 20,
    weights_path: Path | None = None,
) -> SegmentResult:
    model, device = load_model(weights_path)
    img = _to_float_image(image)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    # U-Net 2-es pooling miatt páros méret kell
    h, w = tensor.shape[-2:]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()
    binary = prob >= threshold
    binary = remove_small_objects(binary, min_size=min_size)
    labels = label(binary)
    return SegmentResult(
        probability=prob.astype(np.float32),
        mask=binary.astype(np.uint8),
        labels=labels.astype(np.int32),
        n_objects=int(labels.max()),
        method="unet",
    )


def segment_classical(image: np.ndarray, threshold: float | None = None, min_size: int = 20) -> SegmentResult:
    img = _to_float_image(image)
    cut = float(threshold_otsu(img)) if threshold is None else float(threshold)
    binary = img >= cut
    binary = remove_small_objects(binary, min_size=min_size)
    labels = label(binary)
    return SegmentResult(
        probability=img,
        mask=binary.astype(np.uint8),
        labels=labels.astype(np.int32),
        n_objects=int(labels.max()),
        method="otsu" if threshold is None else "threshold",
    )


def segment_image(
    image: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 20,
    prefer_unet: bool = True,
    weights_path: Path | None = None,
) -> SegmentResult:
    if prefer_unet:
        try:
            return segment_with_unet(image, threshold=threshold, min_size=min_size, weights_path=weights_path)
        except FileNotFoundError:
            pass
    return segment_classical(image, threshold=None, min_size=min_size)


def count_objects(result: SegmentResult) -> list[dict]:
    rows = []
    for region in regionprops(result.labels):
        rows.append(
            {
                "id": int(region.label),
                "area": int(region.area),
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
            }
        )
    return rows
