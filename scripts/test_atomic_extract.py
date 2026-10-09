"""Host filesystem tests for atomic RVC feature caches (not neural CPU fallback)."""
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rvc.train.extract.storage import atomic_save_npy, valid_npy


def main():
    with tempfile.TemporaryDirectory(prefix="xpu_features_") as folder:
        dirpath = pathlib.Path(folder)
        pitch = dirpath / "f0" / "voice.wav.npy"
        feat = dirpath / "embed" / "voice.npy"
        good_pitch = np.array([80, 97, 140], dtype=np.int64)
        good_feature = np.ones((8, 768), dtype=np.float32)

        assert not valid_npy(pitch, 1)
        atomic_save_npy(pitch, good_pitch)
        assert valid_npy(pitch, 1)
        np.testing.assert_array_equal(np.load(pitch, allow_pickle=False), good_pitch)

        atomic_save_npy(feat, good_feature)
        assert valid_npy(feat, 2)
        assert not valid_npy(feat, 1)

        # Real crashes can leave a truncated file. Such a cache must not be
        # trusted as complete input for RVC training.
        feat.write_bytes(b"\x93NUMPY")
        assert not valid_npy(feat, 2)
        atomic_save_npy(feat, good_feature)
        assert valid_npy(feat, 2)

        corrupt = np.full((4, 768), np.nan, dtype=np.float32)
        atomic_save_npy(feat, corrupt)
        assert not valid_npy(feat, 2)
        atomic_save_npy(feat, good_feature)
        assert valid_npy(feat, 2)
        assert not list(dirpath.rglob("*.tmp")), "Temporary files must be atomically renamed"

    print("PASS: XPU RVC pitch and embedding atomic writes, crash recovery and cache validation")


if __name__ == "__main__":
    main()
