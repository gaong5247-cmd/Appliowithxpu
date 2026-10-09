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
        if name == "weight_g":
            name = "parametrizations.weight.original0"
        elif name == "weight_v":
            name = "parametrizations.weight.original1"
        elif name.endswith(".weight_g"):
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
    # Validate BEFORE modifying any model parameters. Loading with
    # strict=False and checking only afterwards risks partially replacing
    # a live voice model when the checkpoint is from another architecture.
    expected = model.state_dict()
    missing_before = sorted(set(expected) - set(converted))
    unexpected_before = sorted(set(converted) - set(expected))
    missing = [k for k in missing_before if not k.startswith("enc_q.")]
    unexpected = [k for k in unexpected_before if not k.startswith("enc_q.")]
    wrong_shapes = [
        (k, tuple(converted[k].shape), tuple(expected[k].shape))
        for k in set(expected).intersection(converted)
        if hasattr(converted[k], "shape")
        and tuple(converted[k].shape) != tuple(expected[k].shape)
    ]
    if missing or unexpected or wrong_shapes:
        raise RuntimeError(
            "RVC voice checkpoint incompatible with this model architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(wrong_shapes)} shape mismatches; "
            f"missing example={missing[:8]}; unexpected example={unexpected[:8]}; "
            f"shapes example={wrong_shapes[:4]}"
        )
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
