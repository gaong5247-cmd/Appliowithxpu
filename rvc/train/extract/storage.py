"""Atomic NumPy cache writes for XPU RVC pitch/content feature extraction.

All functions are host-side disk I/O and do not run neural computation.
"""
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_save_npy(filename, array):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def valid_npy(filename, ndim):
    """Treat truncated, empty, NaN/Inf or wrong-rank feature files as invalid."""
    try:
        arr = np.load(filename, allow_pickle=False, mmap_mode="r")
        if arr.ndim != ndim or arr.size == 0:
            return False
        if arr.dtype.kind not in "fiu":
            return False
        return bool(np.isfinite(arr).all())
    except (OSError, ValueError, TypeError, EOFError):
        return False
