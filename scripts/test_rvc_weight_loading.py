"""Regression for public RVC weight_g/weight_v voice checkpoints.

The host-based test proves that legacy exported tensors are fully reloaded
into modern PyTorch parametrizations, rather than ignored by strict=False.
It is independent of Intel GPU availability.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.utils.parametrizations import weight_norm
from rvc.lib.weights import normalize_rvc_weight_keys, load_rvc_voice_weights


def make_model():
    return weight_norm(torch.nn.Conv1d(2, 3, kernel_size=3, padding=1))


def main():
    torch.manual_seed(1024)
    source = make_model()
    reference = source(torch.randn(1, 2, 12))
    params = {name: t.detach().clone() for name, t in source.state_dict().items()}
    legacy = {
        name.replace(".parametrizations.weight.original0", ".weight_g")
            .replace(".parametrizations.weight.original1", ".weight_v"): t.clone()
        for name, t in params.items()
    }
    assert any(k.endswith(".weight_g") for k in legacy)
    assert any(k.endswith(".weight_v") for k in legacy)

    target = make_model()
    load_rvc_voice_weights(target, legacy)
    for name, expected in params.items():
        assert torch.equal(expected, target.state_dict()[name]), name

    modern = make_model()
    load_rvc_voice_weights(modern, params)
    for name, expected in params.items():
        assert torch.equal(expected, modern.state_dict()[name]), name

    malformed = dict(legacy)
    # The legacy key, not its modern conversion, is actually present.
    malformed.pop("weight_g", None)
    try:
        load_rvc_voice_weights(make_model(), malformed)
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("RVC missing weight norm tensor silently accepted")

    duplicate = dict(legacy)
    duplicate["parametrizations.weight.original0"] = params["parametrizations.weight.original0"]
    try:
        normalize_rvc_weight_keys(duplicate)
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate conflicting RVC weight keys accepted")

    with torch.no_grad():
        assert torch.isfinite(reference).all()
    print("PASS RVC legacy weight_g/v -> parametrization mapping and strict validation")


if __name__ == "__main__":
    main()
