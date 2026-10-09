"""Regression for crash-safe PyTorch model and training checkpoints.

Runs on GitHub Windows host CPU solely to test file/serialization operations.
No neural compute fallback is created for the Intel XPU application.
"""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rvc.train.process.atomic_checkpoint import atomic_torch_save


def main():
    with tempfile.TemporaryDirectory(prefix="rvc_atomic_") as folder:
        path = Path(folder) / "model.pth"
        initial = {"weights": torch.tensor([3.0, 4.0]), "epoch": 10}
        second = {"weights": torch.tensor([6.0, 7.0]), "epoch": 11}
        atomic_torch_save(initial, path)
        assert torch.load(path, weights_only=True)["epoch"] == 10
        original = path.read_bytes()

        with patch("rvc.train.process.atomic_checkpoint.torch.save", side_effect=OSError("disk full")):
            try:
                atomic_torch_save(second, path)
            except OSError as exc:
                assert "disk full" in str(exc)
            else:
                raise AssertionError("Failed checkpoint serialization was hidden")

        assert path.read_bytes() == original, "A failed write destroyed a valid checkpoint!"
        assert not list(Path(folder).glob(".rvc-writing-*"))

        atomic_torch_save(second, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["epoch"] == 11
        assert torch.equal(loaded["weights"], second["weights"])
        assert not list(Path(folder).glob(".rvc-writing-*"))

    print("PASS: atomic G/D and inference .pth saving, disk error recovery and no stale temp files")


if __name__ == "__main__":
    main()
