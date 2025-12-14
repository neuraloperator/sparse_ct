from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Union
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import os
import SimpleITK as sitk
import cv2  # <-- needed for resize

# If you have this in your codebase, keep the import.
# Otherwise you can remove BaseDataModule inheritance and the super() calls.
from src.data.base import BaseDataModule


# ----- CT utils (same logic as older script) -----
class CTTools:
    def __init__(self, mu_water: float = 0.192):
        self.mu_water = mu_water

    def HU2mu(self, hu_img: np.ndarray) -> np.ndarray:
        # μ = μ_water * (HU/1000 + 1)
        return hu_img / 1000.0 * self.mu_water + self.mu_water

    def mu2HU(self, mu_img: np.ndarray) -> np.ndarray:
        return (mu_img - self.mu_water) / self.mu_water * 1000.0


class CTSample(NamedTuple):
    image: torch.Tensor  # [C,H,W]


class AAPMDataset(Dataset):
    """
    Simple .npy image dataset with the SAME preprocessing as the older script:
      - Assume single-channel HU slice.
      - Force spatial size to 256x256 using bicubic interpolation.
      - Convert HU -> μ using CTTools(mu_water=0.192).
      - Return [1,256,256] float32 tensor.
    """
    TARGET_HW = (256, 256)  # (H, W)

    def __init__(self, data_dir: Union[str, pathlib.Path]):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"{self.data_dir} does not exist")
        self.files: List[pathlib.Path] = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.data_dir}")

        self.cttool = CTTools()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> CTSample:
        path = self.files[idx]
        # Load as HU and squeeze like the old dataset
        arr = np.load(path, allow_pickle=False).squeeze()

        # Coerce to 2D single-channel (older script expects a 2D slice)
        if arr.ndim == 3:
            # If a singleton channel exists, drop it.
            if 1 in arr.shape:
                arr = np.squeeze(arr)
            else:
                raise ValueError(f"Expected single-channel CT slice for {path.name}, got {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"Unsupported array shape {arr.shape} for {path.name}; expected 2D single-channel.")

        # Resize to (256,256) with bicubic, like cv2.INTER_CUBIC in the old code.
        H, W = arr.shape
        if (H, W) != self.TARGET_HW:
            # cv2 expects (width, height)
            arr = cv2.resize(arr.astype(np.float32, copy=False),
                             (self.TARGET_HW[1], self.TARGET_HW[0]),
                             interpolation=cv2.INTER_CUBIC)
        else:
            arr = arr.astype(np.float32, copy=False)

        # HU -> μ conversion (identical to older script’s CTTools.HU2mu)
        mu = self.cttool.HU2mu(arr)

        # Return [1,H,W] float32
        tensor = torch.from_numpy(mu).unsqueeze(0).to(torch.float32).contiguous()
        return CTSample(image=tensor)


@dataclass
class AAPMDataModule(BaseDataModule):
    """
    DataModule for AAPM-style folder:
      root/
        train_img/*.npy
        test_img/*.npy

    Splits train_img into train/val with ratio val_ratio (default 0.1).

    Note: Preprocessing matches the older script inside AAPMDataset:
          resize -> 256x256 (bicubic) and HU->μ conversion.
    """
    def __init__(
        self,
        shape,                              # kept for BaseDataModule compatibility
        data_dir: Union[str, pathlib.Path], # directory containing train_img/ and test_img/
        batch_size: int,
        num_workers: int = 8,
        val_ratio: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
    ):
        super().__init__(shape, batch_size)
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"{self.data_dir} does not exist")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        train_dir = self.data_dir / "train_img"
        test_dir  = self.data_dir / "test_img"

        full_train = AAPMDataset(train_dir)
        n_total = len(full_train)

        # Deterministic 90/10 split by order, like the older script
        val_ratio = self.val_ratio if self.val_ratio is not None else 0.1
        n_train = int(n_total * (1.0 - val_ratio))
        n_val = n_total - n_train

        # Indices: first 90% -> train, last 10% -> val
        train_idx = list(range(0, n_train))
        val_idx   = list(range(n_total - n_val, n_total))

        print(f"dataset:{n_total}, training set:{len(train_idx)}, val set:{len(val_idx)}")

        self.train_dataset = Subset(full_train, train_idx)
        self.val_dataset   = Subset(full_train, val_idx)
        self.test_dataset  = AAPMDataset(test_dir)

    def train_dataloader(self):
        print(f'-------> training set: {int(np.ceil(len(self.train_dataset)/self.batch_size))} batches of size {self.batch_size} ({len(self.train_dataset)} samples in total) <-------')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        print(f'-------> validation set: {len(self.val_dataset)} batches of size 1 ({len(self.val_dataset)} samples in total) <-------')
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def test_dataloader(self):
        print(f'-------> test set: {len(self.test_dataset)} batches of size 1 ({len(self.test_dataset)} samples in total) <-------')
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )




def ima2array(ima_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(ima_path))

if __name__ == '__main__':
    # Input your directory to unzipped AAPM16 dataset here, download link: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715
    # Unzip the FD_1mm.zip to full_1mm
    patients_dir = 'Dataset_dir_to_downloaded_AAMP16/Patient_Data/Training_Image_Data/1mm B30/full_1mm/'
    patients = sorted(os.listdir(patients_dir))
    data_len_dict = {}
    for patient in patients:
        patient_dir = os.path.join(patients_dir, patient, 'full_1mm')
        num_slices = len(os.listdir(patient_dir))
        data_len_dict[patient] = num_slices
    # print(f'Data from Patient {patient} contain {num_slices} slices.')

    test_patient = ['L506']
    train_val_patient = []
    num_test = 0
    num_train_val = 0
    for patient in patients:
        if patient not in test_patient:
            train_val_patient.append(patient)
            num_train_val += data_len_dict[patient]
        else:
            num_test += data_len_dict[patient]

    print(f'In total, there are {num_train_val + num_test} slices from {len(patients)} patients.')
    print(f'The divided dataset consists of {num_train_val} train/val image pairs, making up {num_train_val/(num_train_val + num_test) * 100}%.')
    print(f'The divided dataset consists of {num_test} test image pairs, making up {num_test/(num_train_val + num_test) * 100}%.')

    

    target_root = 'data/aapm16'
    target_tag = {
        'train': 'train',
        'test':  'test'
    }
    patients = {
        'train': train_val_patient,
        'test': test_patient
    }
    for type in target_tag:
        target_dir = os.path.join(target_root, f'{target_tag[type]}_img')
        os.makedirs(target_dir, exist_ok=True)
        target_patient = patients[type]
        for patient in target_patient:
            source_dir = os.path.join(patients_dir, patient, 'full_1mm')
            ima_files = sorted([_ for _ in os.listdir(source_dir) if '.IMA' in _])
            for ima_file in ima_files:
                arr_file = ima_file.replace('.IMA', '.npy')
                ima_path = os.path.join(source_dir, ima_file)
                arr_path = os.path.join(target_dir, arr_file)

                if os.path.exists(ima_path) and not os.path.exists(arr_path):
                    arr = ima2array(ima_path)                
                    np.save(arr_path, arr)
                    print('.npy file saved: ', arr_file)