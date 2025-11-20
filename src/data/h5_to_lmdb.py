#!/usr/bin/env python3
# h5_to_lmdb.py (v2)
import argparse
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


def _collect_h5(split_dir: pathlib.Path) -> List[pathlib.Path]:
    return sorted([p for p in split_dir.glob("*.h5") if p.is_file()])


def _np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)  # dtype + shape preserved
    return buf.getvalue()


def _estimate_map_size(files: List[pathlib.Path]) -> int:
    total = 0
    for p in tqdm(files, desc="Estimating map_size", leave=False):
        try:
            with h5py.File(p, "r") as hf:
                d = hf["image"]
                total += int(d.size) * int(np.dtype(d.dtype).itemsize)
        except Exception:
            total += 1 << 20  # conservative cushion per unreadable file
    total = int(total * 1.10)   # np.save header
    total = int(total * 1.25)   # LMDB overhead
    return max(total, 1 << 30)  # >= 1GB


def _prepare_out_path(path: pathlib.Path, subdir: bool, force: bool):
    """
    Ensure the destination does not conflict with subdir mode.
    subdir=False -> path must be a FILE (create file)
    subdir=True  -> path must be a DIRECTORY (create dir)
    """
    if path.exists():
        if force:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            if subdir and path.is_file():
                raise RuntimeError(
                    f"{path} exists as a FILE but --subdir true expects a DIRECTORY. "
                    f"Delete it or pass --force."
                )
            if (not subdir) and path.is_dir():
                raise RuntimeError(
                    f"{path} exists as a DIRECTORY but --subdir false expects a FILE. "
                    f"Delete it or pass --force or re-run with --subdir true."
                )
    # Make parents; for subdir=True, LMDB will create the directory itself if needed
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_split(
    in_dir: pathlib.Path,
    out_db: pathlib.Path,
    enforce_shape: Optional[Tuple[int, int]],
    commit_every: int,
    map_size_gb: Optional[float],
    subdir: bool,
    force: bool,
):
    files = _collect_h5(in_dir)
    if not files:
        raise RuntimeError(f"No .h5 files in {in_dir}")

    _prepare_out_path(out_db, subdir=subdir, force=force)
    map_size = int(map_size_gb * (1 << 30)) if map_size_gb is not None else _estimate_map_size(files)

    # Open env
    env = lmdb.open(
        str(out_db),
        map_size=map_size,
        subdir=subdir,
        readonly=False,
        lock=True,        # writer needs lock
        readahead=True,
        meminit=False,
        max_dbs=1,
    )

    keys: List[bytes] = []
    skipped = 0
    log_path = out_db.with_suffix(".wrong_shape.txt")
    wlog = open(log_path, "w")

    try:
        txn = env.begin(write=True)
        try:
            for idx, p in enumerate(tqdm(files, desc=f"Building {out_db.name}")):
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

                if enforce_shape is not None and img.shape != enforce_shape:
                    skipped += 1
                    wlog.write(f"WRONG_SHAPE\t{p}\tshape={img.shape}\n")
                    continue

                if img.dtype != np.float32:
                    img = img.astype(np.float32, copy=False)

                key = f"{idx:09d}".encode("ascii")
                txn.put(key, _np_to_bytes(img))
                keys.append(key)

                if (idx + 1) % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            # Store manifest
            txn.put(b"__len__", str(len(keys)).encode("ascii"))
            txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit()
        finally:
            # If txn is still active (exceptions), abort it
            try:
                txn.abort()
            except Exception:
                pass
    finally:
        wlog.close()
        env.sync()
        env.close()

    print(f"{out_db}: kept={len(keys)} skipped={skipped} | log={log_path}")


def main():
    ap = argparse.ArgumentParser("Convert existing h5/{train,val,test} into lmdb/*.lmdb")
    ap.add_argument("--h5_root", required=True, help="e.g., dataset/C4Kc/h5")
    ap.add_argument("--out_root", required=True, help="e.g., dataset/C4Kc/lmdb")
    ap.add_argument("--shape", type=int, nargs=2, default=[-1, -1],
                    help="Enforce H W; use -1 -1 to allow any shape (no skip).")
    ap.add_argument("--commit_every", type=int, default=4096, help="Batch size for LMDB commits.")
    ap.add_argument("--map_size_gb", type=float, default=None, help="Override LMDB map size in GB.")
    ap.add_argument("--subdir", type=str, default="false", choices=["true", "false"],
                    help="LMDB layout. false -> single file .lmdb (default). true -> directory LMDB.")
    ap.add_argument("--force", action="store_true",
                    help="Delete existing destination path if it already exists.")
    args = ap.parse_args()

    enforce_shape = None if args.shape == [-1, -1] else (args.shape[0], args.shape[1])
    subdir = (args.subdir.lower() == "true")
    h5_root = pathlib.Path(args.h5_root)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        _write_split(
            in_dir=h5_root / split,
            out_db=out_root / f"{split}.lmdb",
            enforce_shape=enforce_shape,
            commit_every=args.commit_every,
            map_size_gb=args.map_size_gb,
            subdir=subdir,
            force=args.force,
        )


if __name__ == "__main__":
    main()

