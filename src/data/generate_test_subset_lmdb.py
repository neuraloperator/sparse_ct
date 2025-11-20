#!/usr/bin/env python3
# build_test_lmdb.py

import io
import os
import shutil
import pathlib
import pickle
from typing import List, Optional, Tuple

import h5py
import lmdb
import numpy as np
from tqdm import tqdm
import random

# =========================
# CONFIG (edit here)
# =========================
H5_TEST_DIR   = pathlib.Path("/global/homes/p/peterwg/pscratch/aujasvit/multires-ct/dataset/C4KS/h5/test")   # folder containing *.h5
OUT_DB_PATH   = pathlib.Path("/global/homes/p/peterwg/pscratch/aujasvit/multires-ct/dataset/C4KS/lmdb_test_subset_500/test_500.lmdb")  # file or directory path (see SUBDIR)
SUBDIR        = False          # False => single-file .lmdb ; True => directory LMDB
FORCE         = True           # delete OUT_DB_PATH if it exists
NUM_SAMPLES   = 500           # how many samples to keep (prefix of shuffled order)
RNG_SEED      = 1337           # fixed seed => reproducible shuffle
ENFORCE_SHAPE = (512, 512)     # or None to disable shape filtering
COMMIT_EVERY  = 4096
MAP_SIZE_GB   = None           # or float like 8.0 to force mapsize
# =========================

def _collect_h5(split_dir: pathlib.Path) -> List[pathlib.Path]:
    # Canonicalize to make shuffle independent of filesystem order
    return sorted([p for p in split_dir.glob("*.h5") if p.is_file()])

def _np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def _estimate_map_size(files: List[pathlib.Path]) -> int:
    total = 0
    for p in tqdm(files, desc="Estimating map_size", leave=False):
        try:
            with h5py.File(p, "r") as hf:
                d = hf["image"]
                total += int(d.size) * int(np.dtype(d.dtype).itemsize)
        except Exception:
            total += 1 << 20
    total = int(total * 1.10)   # np.save header
    total = int(total * 1.25)   # LMDB overhead
    return max(total, 1 << 30)

def _prepare_out_path(path: pathlib.Path, subdir: bool, force: bool):
    if path.exists():
        if force:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            if subdir and path.is_file():
                raise RuntimeError(f"{path} exists as FILE but subdir=True expects a DIRECTORY. Use FORCE or delete.")
            if (not subdir) and path.is_dir():
                raise RuntimeError(f"{path} exists as DIRECTORY but subdir=False expects a FILE. Use FORCE or set subdir=True.")
    path.parent.mkdir(parents=True, exist_ok=True)

def _write_test_split(
    in_dir: pathlib.Path,
    out_db: pathlib.Path,
    enforce_shape: Optional[Tuple[int, int]],
    commit_every: int,
    map_size_gb: Optional[float],
    subdir: bool,
    force: bool,
    num_samples: int,
    rng_seed: int,
):
    # 1) Collect and canonicalize
    all_files = _collect_h5(in_dir)
    if not all_files:
        raise RuntimeError(f"No .h5 files in {in_dir}")

    # 2) Reproducible shuffle independent of NUM_SAMPLES
    rnd = random.Random(rng_seed)
    files = all_files[:]              # copy
    rnd.shuffle(files)

    # 3) Prepare output
    _prepare_out_path(out_db, subdir=subdir, force=force)

    # mapsize: rough upper bound using all files, safe if we later keep subset
    map_size = int(map_size_gb * (1 << 30)) if map_size_gb is not None else _estimate_map_size(files)

    env = lmdb.open(
        str(out_db),
        map_size=map_size,
        subdir=subdir,
        readonly=False,
        lock=True,
        readahead=True,
        meminit=False,
        max_dbs=1,
    )

    keys: List[bytes] = []
    written = 0
    skipped = 0
    log_path = out_db.with_suffix(".wrong_shape.txt") if not subdir else out_db.parent / (out_db.name + ".wrong_shape.txt")
    wlog = open(log_path, "w")

    try:
        txn = env.begin(write=True)
        try:
            # Iterate the shuffled list; write until we have num_samples valid items
            for idx, p in enumerate(tqdm(files, desc=f"Building {out_db.name if not subdir else out_db} (test only)")):
                if written >= num_samples:
                    break
                try:
                    with h5py.File(p, "r") as hf:
                        img = hf["image"][...]
                except Exception as e:
                    skipped += 1
                    wlog.write(f"READ_FAIL\t{p}\t{repr(e)}\n")
                    continue

                if img.ndim != 2:
                    skipped += 1
                    wlog.write(f"NDIM!=2\t{p}\tshape={img.shape}\n")
                    continue

                if enforce_shape is not None and tuple(img.shape) != tuple(enforce_shape):
                    skipped += 1
                    wlog.write(f"WRONG_SHAPE\t{p}\tshape={img.shape}\n")
                    continue

                if img.dtype != np.float32:
                    img = img.astype(np.float32, copy=False)

                # Use sequential numeric keys => deterministic lexicographic == numeric
                key = f"{written:09d}".encode("ascii")
                txn.put(key, _np_to_bytes(img))
                keys.append(key)
                written += 1

                if (len(keys)) % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            # Manifest
            txn.put(b"__len__", str(len(keys)).encode("ascii"))
            txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit()
        finally:
            try:
                txn.abort()
            except Exception:
                pass
    finally:
        wlog.close()
        env.sync()
        env.close()

    print(f"{out_db}: kept={len(keys)} skipped={skipped} (requested={num_samples}) | log={log_path}")

def main():
    _write_test_split(
        in_dir=H5_TEST_DIR,
        out_db=OUT_DB_PATH,
        enforce_shape=ENFORCE_SHAPE,
        commit_every=COMMIT_EVERY,
        map_size_gb=MAP_SIZE_GB,
        subdir=SUBDIR,
        force=FORCE,
        num_samples=NUM_SAMPLES,
        rng_seed=RNG_SEED,
    )

if __name__ == "__main__":
    main()
