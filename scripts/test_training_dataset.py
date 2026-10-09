"""CI-only test of the actual RVC dataset loader against synthetic training data.

This intentionally uses host tensors on GitHub runners because they have no
Intel GPU. It is NOT an Applio runtime CPU fallback or a speed benchmark.
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.xpu_one_epoch import make_dataset
from rvc.train.utils import HParams
from rvc.train.data_utils import TextAudioLoaderMultiNSFsid, TextAudioCollateMultiNSFsid
import json


def check_rate(sample_rate):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"__ci_rvc_dataset_{sample_rate}_", dir=logs) as folder:
        exp = pathlib.Path(folder)
        make_dataset(exp, sample_rate=sample_rate)
        settings = json.loads((exp / "config.json").read_text(encoding="utf-8"))
        settings["data"]["training_files"] = str(exp / "filelist.txt")
        ds = TextAudioLoaderMultiNSFsid(HParams(**settings).data)
        assert len(ds) == 8, f"Expected 8 genuine RVC dataset items; found {len(ds)}"
        spec, wav, phone, pitch, pitchf, speaker = ds[0]
        fft_bins = settings["data"]["filter_length"] // 2 + 1
        assert spec.ndim == 2 and spec.shape[0] == fft_bins, spec.shape
        assert wav.ndim == 2 and wav.shape[-1] >= 8640, wav.shape
        assert phone.ndim == 2 and phone.shape[-1] == 768, phone.shape
        assert pitch.ndim == 1 and pitchf.ndim == 1
        assert phone.shape[0] == spec.shape[-1] == pitch.shape[0] == pitchf.shape[0]
        batch = TextAudioCollateMultiNSFsid()([ds[0], ds[1]])
        assert len(batch) == 9, len(batch)
        assert batch[0].shape[0] == batch[4].shape[0] == batch[6].shape[0] == 2
        assert torch.isfinite(batch[4]).all()
        assert torch.isfinite(batch[6]).all()
        print(
            f"PASS RVC {sample_rate//1000} kHz synthetic dataset: filelist, label alignment, "
            "real spectrogram loader, batch collation and finite audio (CI HOST ONLY)"
        )


def main():
    for sample_rate in (24000, 32000, 40000, 48000):
        check_rate(sample_rate)


if __name__ == "__main__":
    main()
