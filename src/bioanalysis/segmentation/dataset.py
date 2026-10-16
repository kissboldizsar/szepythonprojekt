from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import generate_cell_image


class SyntheticCellDataset(Dataset):
    def __init__(self, length: int = 64, size: int = 128, seed: int = 0) -> None:
        self.length = length
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = generate_cell_image(size=self.size, seed=self.seed + index)
        image = torch.from_numpy(sample.image).unsqueeze(0)
        mask = torch.from_numpy(sample.mask.astype(np.float32)).unsqueeze(0)
        return image, mask
