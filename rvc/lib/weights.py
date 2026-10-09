"""Legacy RVC weight-normalization checkpoint compatibility.

Public RVC voice files conventionally serialize the old weight_g/weight_v
names. Recent PyTorch weight_norm parametrizations expect original0/original1.
Never silently accept an incompatible or mostly-uninitialized voice model.
"""
from collections import OrderedDict
from collections.abc import Mapping


def normalize_rvc_weight_keys(state_dict):
    if not isinstance(state_dict, Mapping):
        raise TypeError("RVC .pth checkpoint must contain a mapping of tensor names")
    converted = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"Invalid RVC parameter name: {key!r}")
        name = key
        if name.endswith(".weight_g"):
            name = name[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif name.endswith(".weight_v"):
            name = name[: -len(".weight_v")] + ".parametrizations.weight.original1"
        if name in converted:
            raise RuntimeError(
                f"RVC checkpoint contains duplicate weight keys after conversion: {name}"
            )
        converted[name] = value
    return converted


def load_rvc_voice_weights(model, weights):
    """Load RVC inference weights and reject missing/mismatched neural tensors.

    Models can explicitly omit enc_q (training-only posterior encoder), but
    other missing or unexpected tensors usually mean a broken .pth export or
    an incompatible architecture.
    """
    converted = normalize_rvc_weight_keys(weights)
    result = model.load_state_dict(converted, strict=False)
    missing = [name for name in result.missing_keys if not name.startswith("enc_q.")]
    unexpected = [name for name in result.unexpected_keys if not name.startswith("enc_q.")]
    if missing or unexpected:
        raise RuntimeError(
            "RVC voice checkpoint incompatible with this model architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected; "
            f"missing example={missing[:8]}; unexpected example={unexpected[:8]}"
        )
    return model
