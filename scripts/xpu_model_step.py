"""Single *real* RVC generator + discriminator XPU step without audio data.

No pretrained model needed. Runs model architecture, forward, backward and AdamW
on Intel Arc. Designed for user-owned Intel Arc hardware (not GitHub runners).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.configs.config import require_xpu
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator


def main():
    dev = require_xpu(0)
    torch.xpu.set_device(dev)
    config = json.loads(Path("rvc/configs/40000.json").read_text(encoding="utf-8"))
    data, train, model = config["data"], config["train"], config["model"]
    model["spk_embed_dim"] = 1
    print("[RVC step] constructing the real generator and discriminator on", dev)
    net_g = Synthesizer(
        data["filter_length"] // 2 + 1,
        train["segment_size"] // data["hop_length"],
        **model, use_f0=True, sr=data["sample_rate"],
        vocoder="HiFi-GAN", checkpointing=False, randomized=True,
    ).to(dev)
    net_d = MultiPeriodDiscriminator(model["use_spectral_norm"], checkpointing=False).to(dev)
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=1e-4)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=1e-4)

    torch.manual_seed(1234)
    frames = 64
    phone = torch.randn(1, frames, model["text_enc_hidden_dim"], device=dev)
    phone_lengths = torch.full((1,), frames, device=dev, dtype=torch.long)
    pitch = torch.randint(40, 200, (1, frames), device=dev)
    pitchf = torch.rand(1, frames, device=dev) * 250 + 100
    spec = torch.randn(1, data["filter_length"] // 2 + 1, frames, device=dev)
    spec_lengths = phone_lengths.clone()
    speaker = torch.zeros(1, device=dev, dtype=torch.long)

    net_g.train()
    net_d.train()
    start = time.perf_counter()
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        y_hat, *_ = net_g(phone, phone_lengths, pitch, pitchf,
                          spec, spec_lengths, speaker)
        gen_loss = y_hat.float().square().mean()
    gen_loss.backward()
    optim_g.step()
    optim_g.zero_grad(set_to_none=True)
    torch.xpu.synchronize()
    print("GENERATOR FORWARD+BACKWARD+OPTIMIZER OK", y_hat.shape,
          "loss", float(gen_loss.detach().cpu()))

    real = torch.randn_like(y_hat).detach()
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        y_real, y_fake, *_ = net_d(real, y_hat.detach())
        disc_loss = sum(x.float().square().mean() for x in y_real + y_fake)
    disc_loss.backward()
    optim_d.step()
    optim_d.zero_grad(set_to_none=True)
    torch.xpu.synchronize()
    print("DISCRIMINATOR FORWARD+BACKWARD+OPTIMIZER OK",
          "loss", float(disc_loss.detach().cpu()))
    print("XPU REAL RVC MODEL STEP PASS", round(time.perf_counter() - start, 2), "seconds")
    print("This validates a synthetic step, NOT feature extraction or complete training.")


if __name__ == "__main__":
    main()
