"""CI host-only RVC .pth roundtrip through the REAL generator architecture.

Tests export_model -> serialized legacy weight_g/weight_v -> modern Synthesizer
reload. It checks that loaded parameters actually match the originals, not
just that load_state_dict(strict=False) returned without raising.
"""
import copy
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.weights import load_rvc_voice_weights
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import HParams


def main():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / "rvc/configs/24000.json").read_text(encoding="utf-8"))
    model_config = cfg["model"]
    model_config.update(
        inter_channels=32, hidden_channels=32,
        filter_channels=64, n_layers=2, n_heads=2,
        spk_embed_dim=1, gin_channels=16,
        upsample_initial_channel=64,
    )
    hparams = HParams(**cfg)
    generator = Synthesizer(
        cfg["data"]["filter_length"] // 2 + 1,
        32, **model_config, use_f0=True, sr=24000,
        vocoder="HiFi-GAN", randomized=True, checkpointing=False,
    )
    generator.eval()
    base = generator.state_dict()
    with tempfile.TemporaryDirectory(prefix="rvc_roundtrip_") as temp:
        checkpoint = pathlib.Path(temp) / "voice_1e_1s.pth"
        extract_model(
            ckpt=base, sr=24000, name="voice", model_path=str(checkpoint),
            epoch=1, step=1, hps=hparams, vocoder="HiFi-GAN",
        )
        assert checkpoint.is_file(), "RVC export silently failed"
        exported = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert exported["sr"] == 24000
        assert exported["version"] == "v2"
        assert any(k.endswith(".weight_g") for k in exported["weight"])
        assert any(k.endswith(".weight_v") for k in exported["weight"])

        exported["config"][-3] = exported["weight"]["emb_g.weight"].shape[0]
        reloaded = Synthesizer(
            *exported["config"], use_f0=exported.get("f0", 1),
            text_enc_hidden_dim=768, vocoder=exported.get("vocoder", "HiFi-GAN"),
        )
        del reloaded.enc_q
        load_rvc_voice_weights(reloaded, exported["weight"])
        loaded = reloaded.state_dict()

        compared = 0
        for key, reference in base.items():
            if key.startswith("enc_q.") or key not in loaded:
                continue
            # Exports are FP16 for RVC .pth compatibility.
            assert torch.allclose(reference.float(), loaded[key].float(), atol=1e-3, rtol=1e-3), key
            compared += 1
        assert compared > 40, f"Only {compared} RVC tensor names were checked"

    print(f"PASS: {compared} RVC neural tensors exported and reloaded via legacy .pth keys (HOST CI)")


if __name__ == "__main__":
    main()
