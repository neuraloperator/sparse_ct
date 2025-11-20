import io
import os
import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union, NamedTuple
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class CTSample(NamedTuple):
    image: torch.Tensor  # [1, H, W]
    true_image: torch.Tensor


def _bytes_to_np(b: bytes) -> np.ndarray:
    # dtype + shape preserved by np.save during LMDB build
    return np.load(io.BytesIO(b), allow_pickle=False)

class CTTools:
    def __init__(self, mu_water: float = 0.192):
        self.mu_water = mu_water

    def HU2mu(self, hu_img: np.ndarray) -> np.ndarray:
        # μ = μ_water * (HU/1000 + 1)
        return hu_img / 1000.0 * self.mu_water + self.mu_water

    def mu2HU(self, mu_img: np.ndarray) -> np.ndarray:
        return (mu_img - self.mu_water) / self.mu_water * 1000.0


class KitsLMDBDataset(Dataset):
    """
    Fast LMDB reader for single-slice images stored as NPY bytes.
    - Manifest keys: __len__, __keys__
    - Lazy per-worker env/txn
    - Works with BOTH single-file (.lmdb + lock) and directory LMDBs.
    """

    def __init__(self, lmdb_path: Union[str, pathlib.Path]):
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        # Detect layout: file => subdir=False, directory => subdir=True
        self._subdir = os.path.isdir(self.lmdb_path)
        self._keys: List[bytes] = []
        self._length: int = 0
        self._env = None
        self._txn = None
        self._load_manifest()
        self.cttool = CTTools()

    def _open_env(self):
        return lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            max_dbs=1,
            subdir=self._subdir,  # critical for file-vs-dir layout
        )

    def _load_manifest(self) -> None:
        env = self._open_env()
        try:
            with env.begin(write=False) as txn:
                length_b = txn.get(b"__len__")
                keys_b = txn.get(b"__keys__")
                if length_b is None or keys_b is None:
                    raise RuntimeError(f"Manifest missing in {self.lmdb_path} (need __len__ and __keys__).")
                self._length = int(length_b.decode("ascii"))
                self._keys = pickle.loads(keys_b)
        finally:
            env.close()

    def _lazy_open(self) -> None:
        if self._env is None:
            self._env = self._open_env()
            self._txn = self._env.begin(write=False)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> CTSample:
        self._lazy_open()
        key = self._keys[idx]
        val = self._txn.get(key)
        if val is None:
            raise KeyError(f"Key {key!r} not found in {self.lmdb_path}")

        # Decode: expect HxW float32 HU slice (as written by converter)
        arr = _bytes_to_np(val)  # HxW, float32
        if arr.ndim != 2:
            raise ValueError(f"Unsupported array shape {arr.shape} for key {key!r}; expected 2D single-channel.")

        # HU -> μ conversion (same as CTTools.HU2mu in your older script)
        mu = self.cttool.HU2mu(arr.astype(np.float32, copy=False))
        real_mu = mu.copy()
        # Resize to (256,256) using bicubic (cv2.INTER_CUBIC)
        # Note: cv2.resize expects (width, height)
        if mu.shape != (256, 256):
            mu = cv2.resize(mu, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Return [1, H, W] float32 tensor
        tens = torch.from_numpy(mu).unsqueeze(0).to(torch.float32).contiguous()
        true_image = torch.from_numpy(real_mu).unsqueeze(0).to(torch.float32).contiguous()
        return CTSample(image=tens, true_image=true_image)

    def __del__(self):
        try:
            if self._txn is not None:
                self._txn.abort()
            if self._env is not None:
                self._env.close()
        except Exception:
            pass


# If you already have BaseDataModule in your repo, import it.
from src.data.base import BaseDataModule


@dataclass
class KitsLMDBDataModule(BaseDataModule):
    """
    Strictly follows your pattern: ctor stores paths; setup() sets datasets.
    BaseDataModule is responsible for creating DataLoaders.
    """
    def __init__(self, shape, data_dir: Union[str, pathlib.Path], batch_size):
        super().__init__(shape, batch_size)
        self.data_dir = pathlib.Path(data_dir)
        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist"
        self.batch_size = batch_size

    def _resolve_split_path(self, split: str) -> pathlib.Path:
        """
        Accept either:
          - single-file LMDB:  <data_dir>/<split>.lmdb
          - directory LMDB:    <data_dir>/<split>.lmdb/  (lmdb subdir=True)
          - bare:              <data_dir>/<split>
        """
        p = self.data_dir / f"{split}.lmdb"
        if p.exists():
            return p  # works for both file and directory
        b = self.data_dir / split
        if b.exists():
            return b
        raise FileNotFoundError(
            f"Could not find LMDB for split '{split}'. Tried: {p} (file/dir), {b}"
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = KitsLMDBDataset(self._resolve_split_path("train"))
        self.val_dataset   = KitsLMDBDataset(self._resolve_split_path("val"))
        self.test_dataset  = KitsLMDBDataset(self._resolve_split_path("test"))


@dataclass
class KitsLMDBTestDataModule(BaseDataModule):
    """
    Strictly follows your pattern: ctor stores paths; setup() sets datasets.
    BaseDataModule is responsible for creating DataLoaders.
    """
    def __init__(self, shape, data_dir: Union[str, pathlib.Path], batch_size):
        super().__init__(shape, batch_size)
        self.data_dir = pathlib.Path(data_dir)
        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist"
        self.batch_size = batch_size

    def _resolve_split_path(self, split: str) -> pathlib.Path:
        """
        Accept either:
          - single-file LMDB:  <data_dir>/<split>.lmdb
          - directory LMDB:    <data_dir>/<split>.lmdb/  (lmdb subdir=True)
          - bare:              <data_dir>/<split>
        """
        p = self.data_dir / f"{split}.lmdb"
        if p.exists():
            return p  # works for both file and directory
        b = self.data_dir / split
        if b.exists():
            return b
        raise FileNotFoundError(
            f"Could not find LMDB for split '{split}'. Tried: {p} (file/dir), {b}"
        )

    def setup(self, stage: Optional[str] = None):
        self.test_dataset  = KitsLMDBDataset(self._resolve_split_path("test_500"))
