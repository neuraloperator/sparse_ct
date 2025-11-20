from dataclasses import dataclass
import pickle
from typing import List, NamedTuple, Optional, Union
import pydicom
import torch
import pathlib
import os
from torch.utils.data import Dataset
import lmdb
import numpy as np
from tqdm import tqdm
import h5py
from src.data.base import BaseDataModule


class CTSample(NamedTuple):
    image: torch.tensor


def circular_crop(image, fill_value=0):
    """
    Mask to the inscribed circle then center-crop to square,
    returning just the final image.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image.
    fill_value : scalar
        Value to put outside the circle (default 0).

    Returns
    -------
    final : ndarray, shape (min(H,W), min(H,W)) or (..., C)
        The masked & square-cropped image.
    """
    # dimensions and circle parameters
    H, W = image.shape[:2]
    side = min(H, W)
    r = side // 2
    cy, cx = H // 2, W // 2

    # build mask
    Y, X = np.ogrid[:H, :W]
    dist2 = (Y - cy)**2 + (X - cx)**2
    outside = dist2 > r**2

    # warn if discarding nonzero pixels
    # if np.any(image[outside]):
    #     warn('Masking: image must be zero outside the reconstruction circle')

    # apply mask
    masked = image.copy()
    if image.ndim == 2:
        masked[outside] = fill_value
    else:
        masked[outside, :] = fill_value

    # center-crop to square
    excess_h, excess_w = H - side, W - side
    h0 = int(np.ceil(excess_h / 2))
    w0 = int(np.ceil(excess_w / 2))

    final = masked[
        h0: h0 + side,
        w0: w0 + side,
        ...  # preserves channels if present
    ]
    return final


@dataclass
class KitsDataset(Dataset):
    """
    PyTorch Dataset for single-slice .h5 files.
    Expects data_dir to contain only .h5 files (plus optionally length.txt).
    """
    data_dir: pathlib.Path

    def __init__(self, data_dir: Union[str, pathlib.Path]):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        assert self.data_dir.exists(
        ), f"Data directory {self.data_dir} does not exist."

        # collect all .h5 slice files
        # adjust pattern if your files live in subfolders
        self.slices: List[pathlib.Path] = sorted(self.data_dir.glob("*.h5"))
        if not self.slices:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> CTSample:
        h5_path = self.slices[idx]
        with h5py.File(h5_path, "r") as hf:
            # read datasets and convert to torch.Tensor
            image = hf["image"][...]
            # apply circular crop and mask
            # image = circular_crop(image, fill_value=0)
            image = torch.from_numpy(image)
        return CTSample(
            image=image.unsqueeze(0),
        )


@dataclass
class KitsDataModule(BaseDataModule):
    def __init__(self, shape, data_dir: Union[str, pathlib.Path], batch_size):
        super().__init__(shape, batch_size)
        self.data_dir = pathlib.Path(data_dir)
        assert self.data_dir.exists(
        ), f"Data directory {self.data_dir} does not exist"
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = KitsDataset(self.data_dir / "train")
        self.val_dataset = KitsDataset(self.data_dir / "val")
        self.test_dataset = KitsDataset(self.data_dir / "test")



#     def decide_split(dataset_path="dataset/C4KS/", num_patients=10):
#         np.random.seed(42)

#         all_patients = np.arange(0, num_patients, dtype=np.int32)

#         np.random.shuffle(all_patients)
#         all_patients = torch.tensor(all_patients, dtype=torch.int32)
#         train_num = int(0.7 * len(all_patients))
#         val_num = int(0.15 * len(all_patients))
#         test_num = len(all_patients) - train_num - val_num
#         train_patients = all_patients[:train_num]
#         val_patients = all_patients[train_num:train_num + val_num]
#         test_patients = all_patients[train_num + val_num:]
#         split = {
#             "train": train_patients,
#             "val": val_patients,
#             "test": test_patients
#         }

#         torch.save(split, dataset_path + "/split.pt")

#     if not os.path.exists("dataset/C4KS/split.pt"):
#         decide_split()

#     root = pathlib.Path(root_dir)

#     train_lmdb_path = "dataset/C4KS/train.lmdb"
#     val_lmdb_path = "dataset/C4KS/val.lmdb"
#     test_lmdb_path = "dataset/C4KS/test.lmdb"

#     split = torch.load("dataset/C4KS/split.pt")

#     if not os.path.exists(train_lmdb_path):
#         env = lmdb.open(train_lmdb_path, map_size=100*(1 << 30))
#         with env.begin(write=True) as txn:
#             idx = 0
#             # ✅ added tqdm
#             for patient in tqdm(split["train"], desc="Generating Train LMDB"):
#                 patient = int(patient.item())
#                 patient_path = root / f"KiTS-{patient:05d}"

#                 for dirpath, dirnames, filenames in os.walk(patient_path):
#                     dirnames[:] = [
#                         d for d in dirnames if "Segmentation" not in d]

#                     for f in filenames:
#                         if f.endswith(".dcm"):
#                             file_path = os.path.join(dirpath, f)
#                             ds = pydicom.dcmread(file_path)
#                             image = ds.pixel_array

#                             image = image.astype(np.float32)
#                             image = image.clip(0, 0.1)
#                             image = image / 0.35

#                             sinogram, theta = radon(image, num_projections=720)

#                             sinogram = torch.tensor(
#                                 sinogram, dtype=torch.float32)
#                             theta = torch.tensor(theta, dtype=torch.float32)
#                             image = torch.tensor(image, dtype=torch.float32)

#                             sample = CTSample(
#                                 image=image,
#                                 full_sinogram=sinogram,
#                                 theta=theta
#                             )

#                             key = f"{idx:08d}".encode("ascii")
#                             txn.put(key, pickle.dumps(sample))
#                             idx += 1
#             txn.put(b"__len__", str(idx).encode("ascii"))
#         env.close()

#     if not os.path.exists(val_lmdb_path):
#         env = lmdb.open(val_lmdb_path, 100*(1 << 30))
#         with env.begin(write=True) as txn:
#             idx = 0
#             for patient in tqdm(split["val"], desc="Generating Val LMDB"):  # ✅ added tqdm
#                 patient = int(patient.item())
#                 patient_path = root / f"KiTS-{patient:05d}"

#                 for dirpath, dirnames, filenames in os.walk(patient_path):
#                     dirnames[:] = [
#                         d for d in dirnames if "Segmentation" not in d]

#                     for f in filenames:
#                         if f.endswith(".dcm"):
#                             file_path = os.path.join(dirpath, f)
#                             ds = pydicom.dcmread(file_path)
#                             image = ds.pixel_array

#                             image = image.astype(np.float32)
#                             image = image.clip(0, 0.1)
#                             image = image / 0.35

#                             sinogram, theta = radon(image, num_projections=720)

#                             sinogram = torch.tensor(
#                                 sinogram, dtype=torch.float32)
#                             theta = torch.tensor(theta, dtype=torch.float32)
#                             image = torch.tensor(image, dtype=torch.float32)

#                             sample = CTSample(
#                                 image=image,
#                                 full_sinogram=sinogram,
#                                 theta=theta
#                             )

#                             key = f"{idx:08d}".encode("ascii")
#                             txn.put(key, pickle.dumps(sample))
#                             idx += 1
#             txn.put(b"__len__", str(idx).encode("ascii"))
#         env.close()

#     if not os.path.exists(test_lmdb_path):
#         env = lmdb.open(test_lmdb_path, map_size=100*(1 << 30))
#         with env.begin(write=True) as txn:
#             idx = 0
#             for patient in tqdm(split["test"], desc="Generating Test LMDB"):  # ✅ added tqdm
#                 patient = int(patient.item())
#                 patient_path = root / f"KiTS-{patient:05d}"

#                 for dirpath, dirnames, filenames in os.walk(patient_path):
#                     dirnames[:] = [
#                         d for d in dirnames if "Segmentation" not in d]

#                     for f in filenames:
#                         if f.endswith(".dcm"):
#                             file_path = os.path.join(dirpath, f)
#                             ds = pydicom.dcmread(file_path)
#                             image = ds.pixel_array

#                             image = image.astype(np.float32)
#                             image = image.clip(0, 0.1)
#                             image = image / 0.35

#                             sinogram, theta = radon(image, num_projections=720)

#                             sinogram = torch.tensor(
#                                 sinogram, dtype=torch.float32)
#                             theta = torch.tensor(theta, dtype=torch.float32)
#                             image = torch.tensor(image, dtype=torch.float32)

#                             sample = CTSample(
#                                 image=image,
#                                 full_sinogram=sinogram,
#                                 theta=theta
#                             )

#                             key = f"{idx:08d}".encode("ascii")
#                             txn.put(key, pickle.dumps(sample))
#                             idx += 1
#             txn.put(b"__len__", str(idx).encode("ascii"))
#         env.close()


import os
import re
import math
import pathlib
from typing import List, Tuple

import numpy as np
import torch
import h5py
import pydicom
from tqdm import tqdm


def generate_h5_dataset(
    root_dir: str,
    num_patients: int = 210,
    num_train: int = 170,
    out_root: str = "dataset/C4KS/h5",
    seed_split: int = 1092,          # patient split seed
    seed_slices: int = 7737,         # train/val slice assignment + test sampling seed
    val_fraction: float = 0.20,
    test_max_images: int = 10_000,
):
    """
    Builds h5 datasets with:
      - train/test split at the patient level (saved in split.pt)
      - within training patients: slices split into train/val at 80:20 probability
      - slice selection per scan: middle 50% + evenly spaced extremes (5% per end, i.e., 10% total)
      - test set: randomly sample up to 10k images from the remaining patients (no slice filtering)
    """

    # ----------------------------
    # Helpers
    # ----------------------------

    def decide_split(dataset_path: str, num_patients_: int, num_train_: int) -> None:
        np.random.seed(seed_split)
        all_patients = np.arange(num_patients_, dtype=np.int32)
        np.random.shuffle(all_patients)
        all_patients = torch.tensor(all_patients, dtype=torch.int32)

        train_patients = all_patients[:num_train_]
        test_patients = all_patients[num_train_:]

        split = {
            "train_patients": train_patients,  # patients used to create train+val slices
            "test_patients":  test_patients,   # patients used to create test slices
        }

        os.makedirs(dataset_path, exist_ok=True)
        torch.save(split, os.path.join(dataset_path, "split.pt"))

    def numeric_from_name(fname: str) -> int:
        """Fallback numeric key from filename (e.g., '1-154.dcm' -> 154, '001.dcm' -> 1)."""
        m = re.findall(r"\d+", fname)
        if not m:
            return 10**9  # push unparseable names to the end
        try:
            return int(m[-1])
        except ValueError:
            return 10**9

    def sorted_dicom_paths(scan_dir: str) -> List[str]:
        """Return DICOM file paths sorted primarily by InstanceNumber, fallback to numeric filename."""
        all_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".dcm")]
        if not all_files:
            return []

        pairs: List[Tuple[Tuple[int, int], str]] = []
        for f in all_files:
            fp = os.path.join(scan_dir, f)
            inst = None
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                # InstanceNumber is standard for ordering within a series
                inst = int(getattr(ds, "InstanceNumber", None))
            except Exception:
                inst = None

            key1 = inst if inst is not None else 10**9
            key2 = numeric_from_name(f)
            pairs.append(((key1, key2), fp))

        pairs.sort(key=lambda x: x[0])
        return [p for _, p in pairs]

    def select_indices(n: int, ends_fraction_per_side: float = 0.05) -> List[int]:
        """
        Pick middle 50% plus a small, evenly spaced set from both ends.
        - middle: [ceil(0.25*n) .. floor(0.75*n)-1] in 0-based indexing
        - extremes: k indices from the low end and k from the high end, evenly spaced
                    where k = max(1, round(n * ends_fraction_per_side))
        This matches your example for n=100 -> 5 per side (10 total extreme slices).
        """
        if n <= 0:
            return []
        if n == 1:
            return [0]

        # middle 50%
        start = math.floor(0.25 * n)
        end_incl = math.ceil(0.75 * n) - 1
        middle = list(range(start, max(start, min(end_incl + 1, n)))) if end_incl >= start else []

        # remaining extremes
        low_pool = list(range(0, start))
        high_pool = list(range(end_incl + 1, n))

        k_side = max(1, int(round(n * ends_fraction_per_side)))

        def spaced(pool: List[int], k: int) -> List[int]:
            if not pool:
                return []
            k_eff = min(k, len(pool))
            # Evenly spaced indices across the pool (including ends)
            idxs = np.linspace(0, len(pool) - 1, num=k_eff, dtype=int)
            return [pool[i] for i in idxs.tolist()]

        low_sel = spaced(low_pool, k_side)
        high_sel = spaced(high_pool, k_side)

        chosen = sorted(set(middle + low_sel + high_sel))
        return chosen

    # ----------------------------
    # Ensure split exists / load
    # ----------------------------
    split_dir = out_root
    if not os.path.exists(os.path.join(split_dir, "split.pt")):
        decide_split(split_dir, num_patients, num_train)

    root_path = pathlib.Path(root_dir)
    split = torch.load(os.path.join(split_dir, "split.pt"))
    train_patients = [int(x.item()) for x in split["train_patients"]]
    test_patients = [int(x.item()) for x in split["test_patients"]]

    # ----------------------------
    # Prepare output folders
    # ----------------------------
    train_out = pathlib.Path(split_dir) / "train"
    val_out   = pathlib.Path(split_dir) / "val"
    test_out  = pathlib.Path(split_dir) / "test"

    for p in [train_out, val_out, test_out]:
        p.mkdir(parents=True, exist_ok=True)

    # RNG for per-slice assignment and test sampling
    rng = np.random.default_rng(seed_slices)

    # ----------------------------
    # Build train/val from training patients
    # ----------------------------
    train_count = 0
    val_count = 0
    skipped_count = 0

    for patient in tqdm(train_patients, desc="Building train/val from training patients"):
        patient_path = root_path / f"KiTS-{patient:05d}"

        # Walk each scan folder under this patient
        for dirpath, dirnames, filenames in os.walk(patient_path, topdown=True):
            # Skip segmentation folders
            dirnames[:] = [d for d in dirnames if "Segmentation" not in d]

            # If this directory contains DICOMs, treat it as one scan
            scan_dcms = [f for f in filenames if f.lower().endswith(".dcm")]
            if not scan_dcms:
                continue

            scan_dir = dirpath
            dcm_paths = sorted_dicom_paths(scan_dir)
            n = len(dcm_paths)
            if n == 0:
                continue

            sel_idx = select_indices(n, ends_fraction_per_side=0.05)

            # Save selected slices, randomly assigning each to train or val
            local_idx = 0
            for i in sel_idx:
                dcm_fp = dcm_paths[i]
                try:
                    ds = pydicom.dcmread(dcm_fp)
                    image = ds.pixel_array.astype(np.float32)
                except Exception:
                    skipped_count += 1
                    continue

                if rng.random() < (1.0 - val_fraction):
                    # TRAIN
                    out_fp = train_out / f"{patient:05d}-{local_idx}.h5"
                    with h5py.File(out_fp, "w") as hf:
                        hf.create_dataset("image", data=image, compression="gzip")
                    train_count += 1
                else:
                    # VAL
                    out_fp = val_out / f"{patient:05d}-{local_idx}.h5"
                    with h5py.File(out_fp, "w") as hf:
                        hf.create_dataset("image", data=image, compression="gzip")
                    val_count += 1

                local_idx += 1

    with open(train_out / "length.txt", "w") as f:
        f.write(str(train_count))
    with open(val_out / "length.txt", "w") as f:
        f.write(str(val_count))

    # ----------------------------
    # Build test by sampling up to 10k slices from test patients (no filtering)
    # ----------------------------
    # First collect all DICOM file paths from test patients
    test_candidates: List[Tuple[int, str]] = []  # (patient_id, dcm_path)
    for patient in tqdm(test_patients, desc="Collecting test candidates"):
        patient_path = root_path / f"KiTS-{patient:05d}"
        for dirpath, dirnames, filenames in os.walk(patient_path, topdown=True):
            dirnames[:] = [d for d in dirnames if "Segmentation" not in d]
            for fname in filenames:
                if fname.lower().endswith(".dcm"):
                    test_candidates.append((patient, os.path.join(dirpath, fname)))

    if len(test_candidates) == 0:
        sampled = []
    else:
        k = min(test_max_images, len(test_candidates))
        idxs = rng.choice(len(test_candidates), size=k, replace=False).tolist()
        sampled = [test_candidates[i] for i in idxs]

    test_count = 0
    for patient, dcm_fp in tqdm(sampled, desc=f"Saving test (up to {test_max_images})"):
        try:
            ds = pydicom.dcmread(dcm_fp)
            image = ds.pixel_array.astype(np.float32)
        except Exception:
            skipped_count += 1
            continue

        # Name by patient + running count for uniqueness
        out_fp = test_out / f"{patient:05d}-{test_count}.h5"
        with h5py.File(out_fp, "w") as hf:
            hf.create_dataset("image", data=image, compression="gzip")
        test_count += 1

    with open(test_out / "length.txt", "w") as f:
        f.write(str(test_count))

    print(
        f"Done. train={train_count}, val={val_count}, test={test_count}, skipped={skipped_count}\n"
        f"Splits stored in {split_dir}/split.pt"
    )


if __name__ == "__main__":
    # print("Dataset generation completed.")
    generate_h5_dataset()
    print("HDF5 dataset generation completed.")