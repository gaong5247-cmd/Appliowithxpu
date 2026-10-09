"""Real one-epoch Intel XPU RVC training integration test with generated data.

Runs the ACTUAL rvc/train/train.py subprocess and checks G/D checkpoints.
No downloaded audio, pretrained voice, or user data is touched. This is only
for user hardware, not GitHub hosted runners without an Intel Arc GPU.
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import time
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from rvc.configs.config import require_xpu


def make_dataset(exp):
    sr = 24000
    for folder in ("sliced_audios", "f0", "f0_voiced", "extracted"):
        (exp / folder).mkdir(parents=True, exist_ok=True)

    config = json.loads((ROOT / "rvc/configs/24000.json").read_text(encoding="utf-8"))
    (exp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (exp / "model_info.json").write_text(
        json.dumps({"speakers_id": 1, "embedder_model": "contentvec"}),
        encoding="utf-8",
    )

    rng = np.random.default_rng(1234)
    rows = []
    n = sr  # 1 second -> about 100 training frames; longer than 36-frame segment
    samples = np.arange(n, dtype=np.float64) / sr
    frames = 100
    for i in range(8):
        name = f"0_xpu_{i:03}"
        freq = 140.0 + i * 12
        audio = (
            0.06 * np.sin(2 * math.pi * freq * samples)
            + rng.normal(0, 0.002, n)
        ).astype(np.float32)
        wav = exp / "sliced_audios" / f"{name}.wav"
        feat = exp / "extracted" / f"{name}.npy"
        coarse = exp / "f0" / f"{name}.wav.npy"
        fine = exp / "f0_voiced" / f"{name}.wav.npy"

        sf.write(wav, audio, sr, subtype="PCM_16")
        # RVC doubles the (50, 768) content feature sequence to 100 frames.
        embeddings = rng.normal(0, 0.1, (frames // 2, 768)).astype(np.float32)
        np.save(feat, embeddings, allow_pickle=False)
        np.save(coarse, np.full(frames, 110, dtype=np.int64), allow_pickle=False)
        np.save(fine, np.full(frames, freq, dtype=np.float32), allow_pickle=False)
        rows.append("|".join(
            [p.relative_to(ROOT).as_posix() for p in (wav, feat, coarse, fine)] + ["0"]
        ))

    (exp / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"[DATASET] Created {len(rows)} synthetic 24 kHz / 1s clips at {exp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep generated files after success")
    args = parser.parse_args()

    device = require_xpu(0)
    print(f"[XPU] GPU {torch.xpu.get_device_name(0)} - running actual 1-epoch trainer", flush=True)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    exp = pathlib.Path(tempfile.mkdtemp(prefix="__xpu_one_epoch_", dir=logs))
    try:
        make_dataset(exp)
        # Same exact argv structure as core.run_train_script() in the GUI.
        command = [
            sys.executable, str(ROOT / "rvc/train/train.py"),
            exp.name, "1", "1", "", "", "0", "1", "24000",
            "True", "True", "False", "False", "HiFi-GAN", "False",
        ]
        print("[RUN] Full RVC training subprocess, 8 batches, XPU-only neural compute", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"RVC 1-epoch training FAILED exit={result.returncode}; "
                "check the traceback above and preserve this experiment directory"
            )
        gen = sorted(exp.glob("G_*.pth"))
        disc = sorted(exp.glob("D_*.pth"))
        if not gen or not disc:
            raise RuntimeError("Training returned success but a G or D checkpoint is missing")
        g = torch.load(gen[-1], map_location="cpu", weights_only=True)
        for name, tensor in g["model"].items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Generated G checkpoint contains NaN/Inf parameter: {name}")
        print(
            f"[PASS] Real XPU 1-epoch subprocess + G/D checkpoints in "
            f"{time.perf_counter() - start:.1f} seconds",
            flush=True,
        )
    except BaseException:
        print(f"[PRESERVED] Diagnostic experiment remains at: {exp}", flush=True)
        raise
    else:
        if args.keep:
            print(f"[KEPT] Diagnostic data/checkpoints: {exp}")
        else:
            shutil.rmtree(exp)
            print("[CLEAN] Only generated temporary diagnostic experiment removed")


if __name__ == "__main__":
    main()
