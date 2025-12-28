import pathlib
from dataclasses import dataclass
from typing import List, Optional, Union, NamedTuple
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.base import BaseDataModule


class CTSample(NamedTuple):
    image: torch.Tensor       # [1, 256, 256] float32 mu
    true_image: torch.Tensor  # [1, H, W]     float32 mu (original resolution)


class CTTools:
    def __init__(self, mu_water: float = 0.192):
        self.mu_water = mu_water

    def HU2mu(self, hu_img: np.ndarray) -> np.ndarray:
        # μ = μ_water * (HU/1000 + 1)
        return hu_img / 1000.0 * self.mu_water + self.mu_water

    def mu2HU(self, mu_img: np.ndarray) -> np.ndarray:
        return (mu_img - self.mu_water) / self.mu_water * 1000.0


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Accepts HxW, 1xHxW, HxWx1 and returns HxW.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (1, H, W)
        if arr.shape[0] == 1:
            return arr[0]
        # (H, W, 1)
        if arr.shape[-1] == 1:
            return arr[..., 0]
    raise ValueError(f"Unsupported array shape {arr.shape}; expected HxW, 1xHxW, or HxWx1.")


class KitsH5Dataset(Dataset):
    """
    H5 reader matching the LMDB dataset semantics:
      - expects each file to contain dataset key "image"
      - assumes stored values are HU
      - converts HU -> mu
      - returns:
          image      = resized mu to (256, 256) [1,256,256]
          true_image = original mu [1,H,W]
    """
    def __init__(self, data_dir: Union[str, pathlib.Path]):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

        self.slices: List[pathlib.Path] = sorted(self.data_dir.glob("*.h5"))
        if not self.slices:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")

        self.cttool = CTTools()

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> CTSample:
        h5_path = self.slices[idx]
        with h5py.File(h5_path, "r") as hf:
            if "image" not in hf:
                raise KeyError(f"Key 'image' not found in {h5_path}")
            arr = hf["image"][...]

        arr = _ensure_2d(np.asarray(arr))
        hu = arr.astype(np.float32, copy=False)

        # HU -> μ
        mu = self.cttool.HU2mu(hu)
        real_mu = mu.copy()  # original-res μ (true_image)

        # Resize μ to (256,256), bicubic (cv2 expects (W,H))
        if mu.shape != (256, 256):
            mu = cv2.resize(mu, (256, 256), interpolation=cv2.INTER_CUBIC)

        image = torch.from_numpy(mu).unsqueeze(0).to(torch.float32).contiguous()
        true_image = torch.from_numpy(real_mu).unsqueeze(0).to(torch.float32).contiguous()
        return CTSample(image=image, true_image=true_image)


@dataclass
class KitsH5DataModule(BaseDataModule):
    def __init__(self, shape, data_dir: Union[str, pathlib.Path], batch_size: int):
        super().__init__(shape, batch_size)
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = KitsH5Dataset(self.data_dir / "train")
        self.val_dataset   = KitsH5Dataset(self.data_dir / "val")
        self.test_dataset  = KitsH5Dataset(self.data_dir / "test")
