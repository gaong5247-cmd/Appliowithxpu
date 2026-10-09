"""GPU-independent *logic* test for the original RVC G/D architecture.

GitHub-hosted Windows runners have no Intel GPU. This uses small host tensors
ONLY in CI to check model shapes, backward gradients and optimizer behavior.
It is NOT an inference/train fallback available to the Applio app.
"""
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator


def main():
    torch.set_num_threads(2)
    config = json.loads((ROOT / "rvc/configs/24000.json").read_text(encoding="utf-8"))
    model = dict(config["model"])
    # Reduce dimension only for CI footprint; network classes are the same.
    model.update(
        inter_channels=32,
        hidden_channels=32,
        filter_channels=64,
        n_heads=2,
        n_layers=2,
        upsample_initial_channel=64,
        gin_channels=16,
        spk_embed_dim=1,
    )
    data = config["data"]
    frames = 32
    segment_frames = 8
    start = time.perf_counter()

    g = Synthesizer(
        data["filter_length"] // 2 + 1,
        segment_frames,
        **model,
        use_f0=True,
        sr=data["sample_rate"],
        vocoder="HiFi-GAN",
        checkpointing=False,
        randomized=True,
    )
    d = MultiPeriodDiscriminator(False, checkpointing=False)
    optimizer_g = torch.optim.AdamW(g.parameters(), lr=1e-4)
    optimizer_d = torch.optim.AdamW(d.parameters(), lr=1e-4)

    phone = torch.randn(1, frames, model["text_enc_hidden_dim"])
    lengths = torch.tensor([frames])
    pitch = torch.randint(30, 230, (1, frames))
    pitchf = torch.full((1, frames), 160.0)
    spec = torch.randn(1, data["filter_length"] // 2 + 1, frames)
    spk = torch.zeros(1, dtype=torch.long)

    print("[CI-only] RVC generator forward", flush=True)
    y_hat, *_ = g(phone, lengths, pitch, pitchf, spec, lengths, spk)
    assert y_hat.ndim == 3 and y_hat.shape[0] == 1, y_hat.shape
    g_loss = y_hat.float().square().mean()
    g_loss.backward()
    assert any(p.grad is not None for p in g.parameters())
    optimizer_g.step()
    optimizer_g.zero_grad(set_to_none=True)
    print("[CI-only] Generator backward + AdamW PASS", tuple(y_hat.shape), flush=True)

    print("[CI-only] Discriminator forward", flush=True)
    actual = torch.randn_like(y_hat).detach()
    real_out, fake_out, *_ = d(actual, y_hat.detach())
    d_loss = sum(y.float().square().mean() for y in real_out + fake_out)
    d_loss.backward()
    assert any(p.grad is not None for p in d.parameters())
    optimizer_d.step()
    print("[CI-only] Discriminator backward + AdamW PASS", flush=True)

    from rvc.lib.algorithm.commons import slice_segments
    seq = torch.randn(3, 5, 48, requires_grad=True)
    chunks = slice_segments(seq, torch.tensor([0, 8, 17]), 12, dim=3)
    chunks.square().sum().backward()
    assert seq.grad is not None

    print(f"PASS RVC forward/backward/optimizer contract in {time.perf_counter()-start:.2f}s (HOST ONLY, NOT INTEL XPU)")


if __name__ == "__main__":
    main()
