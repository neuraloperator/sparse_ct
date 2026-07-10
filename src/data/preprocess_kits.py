from __future__ import annotations

import os
import re
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pydicom
import torch
from tqdm import tqdm

# =========================================================
# USER CONFIG (edit these 3 paths)
# =========================================================
RAW_DICOM_DIR = Path(os.environ.get("KITS_RAW_DIR", "C4KC-KiTS"))   # original downloaded dataset root (or set KITS_RAW_DIR)
INTER_ORGANIZED_DIR = Path("C4KC-KiTS-organized")       # intermediate patient-wise hierarchy
# =========================================================

# Fixed requirement
TARGET_SHAPE = (512, 512)

# Defaults matching your previous script (you can tweak if needed)
DEFAULT_NUM_PATIENTS = 210
DEFAULT_NUM_TRAIN = 170
DEFAULT_SEED_SPLIT = 1092
DEFAULT_SEED_SLICES = 7737
DEFAULT_VAL_FRACTION = 0.20
DEFAULT_TEST_MAX_IMAGES = 10_000
DEFAULT_ENDS_FRACTION_PER_SIDE = 0.05
H5_DIR = Path(os.environ.get("KITS_DATA_DIR", "data/kits"))   # final h5 dataset root (train/val/test); matches configs/kits.yaml data_dir

# Stage-1 behavior
USE_HARDLINKS = True  # falls back to copy if hardlink fails (e.g., cross-filesystem)


# ----------------------------
# Small utilities
# ----------------------------

def _safe_name(s: object, max_len: int = 80) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w.\-]+", "_", s)  # keep alnum + _ . -
    s = s.strip("._-")
    return (s[:max_len] if s else "Unknown")

def _format_study_date(study_date: str) -> str:
    # DICOM StudyDate typically "YYYYMMDD"
    if isinstance(study_date, str) and len(study_date) == 8 and study_date.isdigit():
        y, m, d = study_date[:4], study_date[4:6], study_date[6:8]
        return f"{y}-{m}-{d}"
    return _safe_name(study_date)

def _patient_tag(pid: str) -> str:
    # For filenames: prefer 5-digit numeric if present, else sanitized string.
    m = re.findall(r"\d+", pid)
    if m:
        try:
            return f"{int(m[-1]):05d}"
        except Exception:
            pass
    return _safe_name(pid, max_len=32)

def _numeric_from_name(fname: str) -> int:
    m = re.findall(r"\d+", fname)
    if not m:
        return 10**9
    try:
        return int(m[-1])
    except ValueError:
        return 10**9

def _sorted_dicom_paths(scan_dir: Path) -> List[Path]:
    all_files = [p for p in scan_dir.iterdir() if p.is_file() and p.name.lower().endswith(".dcm")]
    if not all_files:
        return []

    pairs: List[Tuple[Tuple[int, int], Path]] = []
    for fp in all_files:
        inst: Optional[int] = None
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            inst_val = getattr(ds, "InstanceNumber", None)
            if inst_val is not None:
                inst = int(inst_val)
        except Exception:
            inst = None

        key1 = inst if inst is not None else 10**9
        key2 = _numeric_from_name(fp.name)
        pairs.append(((key1, key2), fp))

    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]

def _select_indices(n: int, ends_fraction_per_side: float) -> List[int]:
    """
    Pick middle 50% plus evenly spaced extremes from both ends.
    middle: [floor(0.25*n) .. ceil(0.75*n)-1]
    extremes: k per side where k=max(1, round(n * ends_fraction_per_side))
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]

    start = math.floor(0.25 * n)
    end_incl = math.ceil(0.75 * n) - 1
    middle = list(range(start, max(start, min(end_incl + 1, n)))) if end_incl >= start else []

    low_pool = list(range(0, start))
    high_pool = list(range(end_incl + 1, n))

    k_side = max(1, int(round(n * ends_fraction_per_side)))

    def spaced(pool: List[int], k: int) -> List[int]:
        if not pool:
            return []
        k_eff = min(k, len(pool))
        idxs = np.linspace(0, len(pool) - 1, num=k_eff, dtype=int)
        return [pool[i] for i in idxs.tolist()]

    chosen = sorted(set(middle + spaced(low_pool, k_side) + spaced(high_pool, k_side)))
    return chosen


# ----------------------------
# Stage 1: organize DICOMs
# ----------------------------

def organize_dicoms(src_dir: Path, dest_root: Path) -> None:
    src_dir = Path(src_dir)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    def iter_dicoms(root: Path):
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(".dcm"):
                    yield Path(dirpath) / f

    moved = 0
    skipped = 0

    for path in tqdm(iter_dicoms(src_dir), desc="Organizing DICOMs", unit="dcm"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            skipped += 1
            continue

        pid = _safe_name(ds.get("PatientID", "UnknownPatient"), max_len=64)
        study_date = _format_study_date(str(ds.get("StudyDate", "UnknownDate")))
        study_desc = _safe_name(ds.get("StudyDescription", "UnknownStudy"))
        series_desc = _safe_name(ds.get("SeriesDescription", "UnknownSeries"))

        study_uid = _safe_name(str(ds.get("StudyInstanceUID", ""))[-6:] or "noUID", max_len=16)
        series_uid = _safe_name(str(ds.get("SeriesInstanceUID", ""))[-6:] or "noUID", max_len=16)

        study_folder = f"{study_date}-{study_desc}-{study_uid}"
        series_folder = f"{series_desc}-{series_uid}"

        dest_dir = dest_root / pid / study_folder / series_folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / path.name
        if dest_path.exists():
            continue

        try:
            if USE_HARDLINKS:
                os.link(str(path), str(dest_path))
            else:
                shutil.copy2(str(path), str(dest_path))
            moved += 1
        except OSError:
            # hardlink can fail across filesystems; fall back to copy
            try:
                shutil.copy2(str(path), str(dest_path))
                moved += 1
            except Exception:
                skipped += 1

    print(f"Organize done: linked/copied={moved}, skipped={skipped}")
    print(f"Organized root: {dest_root}")


# ----------------------------
# Stage 2: build H5 dataset
# ----------------------------

def _discover_patients(root_dir: Path) -> List[str]:
    root_dir = Path(root_dir)
    patients = [p.name for p in root_dir.iterdir() if p.is_dir()]
    patients.sort()
    return patients

def _decide_split(
    out_root: Path,
    patients: List[str],
    num_patients: int,
    num_train: int,
    seed_split: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed_split)
    pats = patients[:num_patients]
    idx = np.arange(len(pats), dtype=np.int32)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    split = {
        "patients": pats,
        "train_patients": [pats[i] for i in train_idx.tolist()],
        "test_patients": [pats[i] for i in test_idx.tolist()],
        "seed_split": int(seed_split),
        "num_patients": int(num_patients),
        "num_train": int(num_train),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(split, str(out_root / "split.pt"))
    return split

def build_h5_dataset(
    organized_root: Path,
    out_root: Path,
    num_patients: int = DEFAULT_NUM_PATIENTS,
    num_train: int = DEFAULT_NUM_TRAIN,
    seed_split: int = DEFAULT_SEED_SPLIT,
    seed_slices: int = DEFAULT_SEED_SLICES,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_max_images: int = DEFAULT_TEST_MAX_IMAGES,
    ends_fraction_per_side: float = DEFAULT_ENDS_FRACTION_PER_SIDE,
) -> None:
    organized_root = Path(organized_root)
    out_root = Path(out_root)

    patients_all = _discover_patients(organized_root)
    if not patients_all:
        raise RuntimeError(f"No patient folders found under: {organized_root}")

    num_patients_eff = min(num_patients, len(patients_all))
    num_train_eff = min(num_train, num_patients_eff)

    split_path = out_root / "split.pt"
    if split_path.exists():
        split = torch.load(str(split_path))
        train_patients = list(split["train_patients"])
        test_patients = list(split["test_patients"])
    else:
        split = _decide_split(
            out_root=out_root,
            patients=patients_all,
            num_patients=num_patients_eff,
            num_train=num_train_eff,
            seed_split=seed_split,
        )
        train_patients = split["train_patients"]
        test_patients = split["test_patients"]

    train_out = out_root / "train"
    val_out = out_root / "val"
    test_out = out_root / "test"
    for p in (train_out, val_out, test_out):
        p.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed_slices)

    train_count = 0
    val_count = 0
    skipped_count = 0

    for patient_id in tqdm(train_patients, desc="Building train/val", unit="patient"):
        patient_path = organized_root / patient_id
        if not patient_path.exists():
            skipped_count += 1
            continue

        per_patient_idx = 0  # ensure uniqueness across ALL series for this patient

        for dirpath, dirnames, filenames in os.walk(patient_path, topdown=True):
            dirnames[:] = [d for d in dirnames if "Segmentation" not in d]

            has_dcms = any(f.lower().endswith(".dcm") for f in filenames)
            if not has_dcms:
                continue

            scan_dir = Path(dirpath)
            dcm_paths = _sorted_dicom_paths(scan_dir)
            n = len(dcm_paths)
            if n == 0:
                continue

            sel_idx = _select_indices(n, ends_fraction_per_side=ends_fraction_per_side)

            for i in sel_idx:
                dcm_fp = dcm_paths[i]
                try:
                    ds = pydicom.dcmread(str(dcm_fp), force=True)
                    image = ds.pixel_array.astype(np.float32)
                except Exception:
                    skipped_count += 1
                    continue

                tag = _patient_tag(patient_id)
                if rng.random() < (1.0 - val_fraction):
                    out_fp = train_out / f"{tag}-{per_patient_idx}.h5"
                    split_name = "train"
                else:
                    out_fp = val_out / f"{tag}-{per_patient_idx}.h5"
                    split_name = "val"

                try:
                    with h5py.File(out_fp, "w") as hf:
                        hf.create_dataset("image", data=image, compression="gzip")
                        hf.attrs["patient_id"] = patient_id
                        hf.attrs["source_dicom"] = str(dcm_fp)
                    if split_name == "train":
                        train_count += 1
                    else:
                        val_count += 1
                    per_patient_idx += 1
                except Exception:
                    skipped_count += 1

    (train_out / "length.txt").write_text(str(train_count))
    (val_out / "length.txt").write_text(str(val_count))

    # --- Test: sample up to test_max_images from test patients (no slice filtering) ---
    test_candidates: List[Tuple[str, Path]] = []
    for patient_id in tqdm(test_patients, desc="Collecting test candidates", unit="patient"):
        patient_path = organized_root / patient_id
        if not patient_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(patient_path, topdown=True):
            dirnames[:] = [d for d in dirnames if "Segmentation" not in d]
            for fname in filenames:
                if fname.lower().endswith(".dcm"):
                    test_candidates.append((patient_id, Path(dirpath) / fname))

    if test_candidates:
        k = min(test_max_images, len(test_candidates))
        idxs = rng.choice(len(test_candidates), size=k, replace=False).tolist()
        sampled = [test_candidates[i] for i in idxs]
    else:
        sampled = []

    test_count = 0
    for patient_id, dcm_fp in tqdm(sampled, desc="Saving test", unit="slice"):
        try:
            ds = pydicom.dcmread(str(dcm_fp), force=True)
            image = ds.pixel_array.astype(np.float32)
        except Exception:
            skipped_count += 1
            continue

        tag = _patient_tag(patient_id)
        out_fp = test_out / f"{tag}-{test_count}.h5"
        try:
            with h5py.File(out_fp, "w") as hf:
                hf.create_dataset("image", data=image, compression="gzip")
                hf.attrs["patient_id"] = patient_id
                hf.attrs["source_dicom"] = str(dcm_fp)
            test_count += 1
        except Exception:
            skipped_count += 1

    (test_out / "length.txt").write_text(str(test_count))

    print(
        f"H5 done. train={train_count}, val={val_count}, test={test_count}, skipped={skipped_count}\n"
        f"Splits stored in {out_root}/split.pt"
    )


# ----------------------------
# Stage 3: move wrong shapes
# ----------------------------

def _get_h5_image_shape(h5path: Path) -> Optional[Tuple[int, ...]]:
    try:
        with h5py.File(h5path, "r") as hf:
            if "image" in hf:
                return tuple(hf["image"].shape)
            for k in hf.keys():
                if isinstance(hf[k], h5py.Dataset):
                    return tuple(hf[k].shape)
    except Exception as e:
        print(f"Warning: failed to read {h5path}: {e}", file=sys.stderr)
    return None

def _ensure_unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def move_wrong_shapes_and_refresh_lengths(data_root: Path) -> None:
    data_root = Path(data_root)

    splits = ["train", "val", "test"]
    wrong_root = data_root / "wrong_shape"

    moved_counts = {s: 0 for s in splits}
    scanned_counts = {s: 0 for s in splits}
    skipped_counts = {s: 0 for s in splits}

    for split in splits:
        split_dir = data_root / split
        dest_dir = wrong_root / split
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue

        for p in split_dir.rglob("*.h5"):
            scanned_counts[split] += 1
            shape = _get_h5_image_shape(p)
            if shape is None:
                skipped_counts[split] += 1
                continue

            if tuple(shape) != TARGET_SHAPE:
                dest_path = _ensure_unique_dest(dest_dir / p.name)
                try:
                    shutil.move(str(p), str(dest_path))
                    moved_counts[split] += 1
                except Exception as e:
                    skipped_counts[split] += 1
                    print(f"Error moving {p} -> {dest_path}: {e}", file=sys.stderr)

    # Refresh length.txt (after moving)
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        n = sum(1 for _ in split_dir.rglob("*.h5"))
        (split_dir / "length.txt").write_text(str(n))

    print("\n=== Wrong-shape summary ===")
    for s in splits:
        print(f"{s}: scanned={scanned_counts[s]}, moved={moved_counts[s]}, skipped={skipped_counts[s]}")
    print(f"Moved files are in: {wrong_root}")
    print("length.txt refreshed for train/val/test.")


# ----------------------------
# Main
# ----------------------------

def main():
    # Minimal CLI override (optional). If you don’t care, just edit USER CONFIG and run.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=str(RAW_DICOM_DIR))
    ap.add_argument("--organized", type=str, default=str(INTER_ORGANIZED_DIR))
    ap.add_argument("--h5", type=str, default=str(H5_DIR))

    ap.add_argument("--skip_organize", action="store_true")
    ap.add_argument("--skip_h5", action="store_true")
    ap.add_argument("--skip_shape_filter", action="store_true")

    # keep your old defaults available
    ap.add_argument("--num_patients", type=int, default=DEFAULT_NUM_PATIENTS)
    ap.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    ap.add_argument("--seed_split", type=int, default=DEFAULT_SEED_SPLIT)
    ap.add_argument("--seed_slices", type=int, default=DEFAULT_SEED_SLICES)
    ap.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--test_max_images", type=int, default=DEFAULT_TEST_MAX_IMAGES)

    args = ap.parse_args()

    raw = Path(args.raw)
    organized = Path(args.organized)
    h5root = Path(args.h5)

    if not args.skip_organize:
        organize_dicoms(raw, organized)

    if not args.skip_h5:
        build_h5_dataset(
            organized_root=organized,
            out_root=h5root,
            num_patients=args.num_patients,
            num_train=args.num_train,
            seed_split=args.seed_split,
            seed_slices=args.seed_slices,
            val_fraction=args.val_fraction,
            test_max_images=args.test_max_images,
            ends_fraction_per_side=DEFAULT_ENDS_FRACTION_PER_SIDE,
        )

    if not args.skip_shape_filter:
        move_wrong_shapes_and_refresh_lengths(h5root)

if __name__ == "__main__":
    main()
