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


def make_dataset(exp, sample_rate=24000):
    sr = int(sample_rate)
    if sr not in (24000, 32000, 40000, 48000):
        raise ValueError(f"Unsupported sample rate: {sr}")
    for folder in ("sliced_audios", "f0", "f0_voiced", "extracted"):
        (exp / folder).mkdir(parents=True, exist_ok=True)

    config = json.loads((ROOT / f"rvc/configs/{sr}.json").read_text(encoding="utf-8"))
    (exp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (exp / "model_info.json").write_text(
        json.dumps({"speakers_id": 1, "embedder_model": "contentvec"}),
        encoding="utf-8",
    )

    rng = np.random.default_rng(1234)
    rows = []
    n = sr  # 1-second clips; no personal data used
    samples = np.arange(n, dtype=np.float64) / sr
    frames = sr // config["data"]["hop_length"]
    assert frames == 100, "RVC synthetic feature alignment requires 100 fps"
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
    print(f"[DATASET] Created {len(rows)} synthetic {sr//1000} kHz / 1s clips at {exp}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep generated files after success")
    parser.add_argument(
        "--sample-rate", type=int, default=24000,
        choices=(24000, 32000, 40000, 48000),
        help="Exercise the real RVC configuration for this sample rate",
    )
    args = parser.parse_args()

    device = require_xpu(0)
    print(f"[XPU] GPU {torch.xpu.get_device_name(0)} - running actual 1-epoch trainer", flush=True)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    exp = pathlib.Path(tempfile.mkdtemp(prefix="__xpu_one_epoch_", dir=logs))
    try:
        make_dataset(exp, sample_rate=args.sample_rate)
        # Same exact argv structure as core.run_train_script() in the GUI.
        command = [
            sys.executable, str(ROOT / "rvc/train/train.py"),
            exp.name, "1", "1", "", "", "0", "1", str(args.sample_rate),
            "True", "True", "False", "False", "HiFi-GAN", "False",
        ]
        print(f"[RUN] Full {args.sample_rate} Hz RVC training subprocess, 8 batches, XPU-only neural compute", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"RVC 1-epoch training FAILED exit={result.returncode}; "
                "check the traceback above and preserve this experiment directory"
            )
        gen = sorted(exp.glob("G_*.pth"))
        disc = sorted(exp.glob("D_*.pth"))
        exported = sorted(exp.glob(f"{exp.name}_*e_*s.pth"))
        if not (gen and disc and exported):
            raise RuntimeError(
                "Training returned exit=0 but checkpoint/export is missing: "
                f"G={bool(gen)} D={bool(disc)} inference_pth={bool(exported)}"
            )
        for label, path in (("G", gen[-1]), ("D", disc[-1])):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            for key, tensor in checkpoint["model"].items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(f"{label} checkpoint has NaN/Inf: {key}")
        voice_model = torch.load(exported[-1], map_location="cpu", weights_only=True)
        if not voice_model.get("weight") or voice_model.get("sr") != args.sample_rate:
            raise RuntimeError("Exported inference .pth is invalid or has the wrong sample rate")

        # The real RVC vocoder must also accept the exported checkpoint.
        # This checks GPU synthesis, not just existence of a .pth filename.
        from rvc.lib.algorithm.synthesizers import Synthesizer
        from rvc.lib.weights import load_rvc_voice_weights
        voice_model["config"][-3] = voice_model["weight"]["emb_g.weight"].shape[0]
        model = Synthesizer(
            *voice_model["config"],
            use_f0=voice_model.get("f0", 1),
            text_enc_hidden_dim=768 if voice_model.get("version", "v2") == "v2" else 256,
            vocoder=voice_model.get("vocoder", "HiFi-GAN"),
        )
        del model.enc_q
        load_rvc_voice_weights(model, voice_model["weight"])
        model = model.to(device).float().eval()
        with torch.inference_mode():
            phonemes = torch.randn(1, 50, 768, device=device)
            lengths = torch.tensor([50], dtype=torch.long, device=device)
            pitch = torch.full((1, 50), 120, dtype=torch.long, device=device)
            pitchf = torch.full((1, 50), 180.0, dtype=torch.float32, device=device)
            speaker = torch.zeros(1, dtype=torch.long, device=device)
            audio_out, *_ = model.infer(phonemes, lengths, pitch, pitchf, speaker)
            torch.xpu.synchronize()
            if audio_out.numel() == 0 or not bool(torch.isfinite(audio_out).all()):
                raise RuntimeError("XPU vocoder generated empty or NaN/Inf audio")
            sf.write(
                exp / "rvc_synthetic_inference.wav",
                audio_out[0, 0].float().cpu().numpy(),
                args.sample_rate,
            )
        print("[PASS] Exported .pth loaded and generated finite waveform on Intel XPU", flush=True)
        print(
            f"[PASS] Real XPU 1-epoch subprocess + G/D + usable inference .pth "
            f"in {time.perf_counter() - start:.1f} seconds",
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
