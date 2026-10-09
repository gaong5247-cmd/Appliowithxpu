"""Pure Python CI test for legacy RVC checkpoint key conversion.

This is intentionally independent of PyTorch and Intel hardware so the fast
Windows source-check workflow can detect incorrect key mappings immediately.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rvc.lib.weights import normalize_rvc_weight_keys


def main():
    legacy = {
        "weight_g": 1,
        "weight_v": 2,
        "dec.ups.0.weight_g": 3,
        "dec.ups.0.weight_v": 4,
        "emb_g.weight": 5,
    }
    normalized = normalize_rvc_weight_keys(legacy)
    assert normalized == {
        "parametrizations.weight.original0": 1,
        "parametrizations.weight.original1": 2,
        "dec.ups.0.parametrizations.weight.original0": 3,
        "dec.ups.0.parametrizations.weight.original1": 4,
        "emb_g.weight": 5,
    }, normalized

    try:
        normalize_rvc_weight_keys(
            {"dec.ups.0.weight_g": 1,
             "dec.ups.0.parametrizations.weight.original0": 2}
        )
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Collision was silently accepted")

    try:
        normalize_rvc_weight_keys({1: "not a parameter name"})
    except TypeError:
        pass
    else:
        raise AssertionError("Non-string RVC parameter name accepted")

    print("PASS: nested and root RVC legacy checkpoint weight names + collision checks")


if __name__ == "__main__":
    main()
