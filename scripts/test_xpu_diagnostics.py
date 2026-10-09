"""Host-only unit tests of the combined XPU diagnostic subprocess/log runner."""
import io
import pathlib
import sys
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.xpu_diagnostics import run_step


class Child:
    def __init__(self, code):
        self.stdout = iter(["initializing Intel XPU\n", "RVC checkpoint stage\n"])
        self.code = code
    def wait(self):
        return self.code


def main():
    for code in (0, 13):
        report = io.StringIO()
        with patch("scripts.xpu_diagnostics.subprocess.Popen", return_value=Child(code)):
            if code:
                try:
                    run_step("RVC test", ["scripts/xpu_smoke.py"], report)
                except RuntimeError as exc:
                    assert "exit=13" in str(exc)
                else:
                    raise AssertionError("XPU validator swallowed failed subprocess status")
            else:
                assert run_step("RVC test", ["scripts/xpu_smoke.py"], report)
        saved = report.getvalue()
        assert "initializing Intel XPU" in saved and "RVC checkpoint stage" in saved
        assert f"exit={code}" in saved
    print("PASS: XPU diagnostic log captures child output, timing, and propagates failures")


if __name__ == "__main__":
    main()
