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


    # A malformed RVC file must be rejected BEFORE load_state_dict touches the model.
    from rvc.lib.weights import load_rvc_voice_weights

    class T:
        def __init__(self, *shape):
            self.shape = shape

    class Model:
        def __init__(self):
            self.loads = 0
        def state_dict(self):
            return {"dec.ups.0.parametrizations.weight.original0": T(3, 1, 1),
                    "emb_g.weight": T(1, 256)}
        def load_state_dict(self, state, strict=False):
            self.loads += 1
            return type("Result", (), {"missing_keys": [], "unexpected_keys": []})()

    bad_shape = Model()
    try:
        load_rvc_voice_weights(bad_shape, {
            "dec.ups.0.weight_g": T(4, 1, 1), "emb_g.weight": T(1, 256)})
    except RuntimeError as exc:
        assert "shape mismatches" in str(exc)
    else:
        raise AssertionError("Wrong-shape RVC model accepted")
    assert bad_shape.loads == 0

    missing = Model()
    try:
        load_rvc_voice_weights(missing, {"emb_g.weight": T(1, 256)})
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Missing RVC weights accepted")
    assert missing.loads == 0

    valid = Model()
    load_rvc_voice_weights(valid, {
        "dec.ups.0.weight_g": T(3, 1, 1), "emb_g.weight": T(1, 256)})
    assert valid.loads == 1

    print("PASS: nested and root RVC legacy checkpoint weight names + collision checks")


if __name__ == "__main__":
    main()
