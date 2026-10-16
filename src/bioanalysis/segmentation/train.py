"""U-Net tanítása szintetikus mikroszkópos mintákon."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticCellDataset
from .unet import UNet, bce_dice_loss

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "data" / "models" / "unet_cells.pt"


def train_unet(
    epochs: int = 8,
    batch_size: int = 8,
    samples: int = 48,
    size: int = 128,
    lr: float = 1e-3,
    weights_path: Path | None = None,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        SyntheticCellDataset(length=samples, size=size, seed=42),
        batch_size=batch_size,
        shuffle=True,
    )
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs}  loss={running / max(len(loader), 1):.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"Súlyok mentve: {weights_path}")
    return weights_path


if __name__ == "__main__":
    train_unet()
