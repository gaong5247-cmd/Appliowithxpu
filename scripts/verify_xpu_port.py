"""CI structural checks; does not pretend to execute XPU kernels on hosted runners."""
from pathlib import Path
import ast


CRITICAL = (
    "rvc/configs/config.py",
    "rvc/train/train.py",
    "rvc/train/extract/extract.py",
    "rvc/infer/infer.py",
    "rvc/infer/pipeline.py",
    "rvc/lib/predictors/RMVPE.py",
    "rvc/lib/predictors/FCPE.py",
    "rvc/realtime/pipeline.py",
    "rvc/realtime/core.py",
)

for name in CRITICAL:
    path = Path(name)
    assert path.is_file(), f"Missing original source file: {name}"
    src = path.read_text(encoding="utf-8")
    ast.parse(src, filename=name)
    if name in (
        "rvc/configs/config.py",
        "rvc/train/train.py",
        "rvc/train/extract/extract.py",
        "rvc/infer/infer.py",
        "rvc/infer/pipeline.py",
        "rvc/lib/predictors/FCPE.py",
    ):
        assert "torch.cuda" not in src, f"CUDA runtime still found: {name}"
assert "xpu" in Path("rvc/train/train.py").read_text()
assert "torch.xpu.is_available()" in Path("rvc/configs/config.py").read_text()
req = Path("requirements.txt").read_text()
assert "torch==" not in req
assert "onnxruntime-gpu" not in req
assert "cu128" not in Path("run-install.bat").read_text()
assert Path("UPSTREAM_COMMIT").read_text().strip() == "324f4d89c3e0e8dc0e555a3d53784e3282c16e09"
assert Path("assets/ICON.ico").exists()
assert Path("LICENSE").exists()

# The XPU build has a strict no-CPU-fallback runtime guard, tested without GPU drivers.
import importlib.util
import sys
import types
fake_torch = types.ModuleType("torch")
class FakeXPU:
    enabled = False
    @classmethod
    def is_available(cls):
        return cls.enabled
    @staticmethod
    def device_count():
        return 1
    @staticmethod
    def get_device_name(idx):
        return "Mock Intel Arc"
fake_torch.xpu = FakeXPU
fake_torch.device = lambda name: name
orig = sys.modules.get("torch")
try:
    sys.modules["torch"] = fake_torch
    spec = importlib.util.spec_from_file_location("xpu_config_ci", "rvc/configs/config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.require_xpu()
    except RuntimeError:
        pass
    else:
        raise AssertionError("CPU fallback unexpectedly accepted")
    FakeXPU.enabled = True
    assert mod.require_xpu() == "xpu:0"
finally:
    if orig is None:
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = orig

print("PASS: source presence, AST, GPU-only runtime guards, XPU references and dependency pinning.")
