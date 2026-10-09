"""Host-side GUI orchestration regression tests, not a runtime CPU fallback.

Check that a child exit code, missing .pth weights or failed FAISS index
cannot be misreported as completed GPU training.
"""
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core


def train(fake_status=0):
    return core.run_train_script(
        model_name="xpu_ci", save_every_epoch=10, save_only_latest=True,
        save_every_weights=True, total_epoch=3, sample_rate=40000,
        batch_size=1, gpu=0, pretrained=False, cleanup=False,
        index_algorithm="Auto", shutdown_check=False,
    )


def main():
    with tempfile.TemporaryDirectory(prefix="xpu_ci_core_") as temp:
        model_dir = pathlib.Path(temp) / "xpu_ci"
        model_dir.mkdir()
        with patch.object(core, "logs_path", temp), \
             patch.object(core.subprocess, "run", return_value=SimpleNamespace(returncode=12)):
            failure = train()
            assert "failed" in failure.lower() and "12" in failure, failure

        # Even a zero subprocess exit must not be treated as successful
        # unless all generator/discriminator/export checkpoints exist.
        with patch.object(core, "logs_path", temp), \
             patch.object(core.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            no_model = train()
            assert "missing" in no_model.lower() and "exported=False" in no_model, no_model

        for name in ("G_1.pth", "D_1.pth", "xpu_ci_3e_99s.pth"):
            (model_dir / name).touch()
        with patch.object(core, "logs_path", temp), \
             patch.object(core.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
             patch.object(core, "run_index_script", return_value="Index generation failed: no vectors"):
            index_failure = train()
            assert "index building failed" in index_failure.lower(), index_failure

        with patch.object(core, "logs_path", temp), \
             patch.object(core.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
             patch.object(core, "run_index_script", return_value="Index file for xpu_ci generated successfully."):
            success = train()
            assert "trained successfully" in success.lower(), success

    print("PASS: RVC GUI rejects worker failures, missing weights, failed index; accepts verified result.")


if __name__ == "__main__":
    main()
