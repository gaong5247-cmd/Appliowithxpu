"""CI-only regression for exported RVC .pth model validity and failures.

This tests filesystem/model serialization on host tensors. GPU-only training
remains guarded by require_xpu().
"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rvc.train.utils import HParams
from rvc.train.process.extract_model import extract_model


def main():
    settings = json.loads((ROOT / "rvc/configs/40000.json").read_text(encoding="utf-8"))
    hps = HParams(**settings)
    with tempfile.TemporaryDirectory(prefix="rvc_export_") as folder:
        root = pathlib.Path(folder)
        model_path = root / "voice_3e_50s.pth"
        extract_model(
            ckpt={"enc_p.example.weight": torch.tensor([0.5, 1.25])},
            sr=40000,
            name="voice",
            model_path=str(model_path),
            epoch=3,
            step=50,
            hps=hps,
            vocoder="HiFi-GAN",
        )
        assert model_path.exists()
        exported = torch.load(model_path, map_location="cpu", weights_only=True)
        assert exported["sr"] == 40000
        assert exported["weight"]["enc_p.example.weight"].dtype == torch.float16
        assert exported["model_name"] == "voice"
        # If saving fails, callers MUST receive an exception.
        blocked = root / "cant_save_here.pth"
        blocked.mkdir()
        try:
            extract_model(
                ckpt={"x": torch.tensor([1.0])},
                sr=40000,
                name="voice",
                model_path=str(blocked),
                epoch=3, step=50, hps=hps, vocoder="HiFi-GAN",
            )
        except Exception:
            pass
        else:
            raise AssertionError("RVC exported .pth save failure was swallowed")

    print("PASS: RVC 40k .pth export, template config fallback, failing save propagation")


if __name__ == "__main__":
    main()
