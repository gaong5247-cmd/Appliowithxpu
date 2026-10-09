"""Intel Arc AdamW optimizer benchmark. All tensor operations on XPU.

Opt-in foreach costs more memory. Run on YOUR hardware and only enable the
environment variable if it clearly improves the comparable benchmark.
"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rvc.configs.config import require_xpu


def measure(foreach):
    device = require_xpu(0)
    torch.manual_seed(10)
    params = [
        torch.nn.Parameter(torch.randn((192, 192), device=device))
        for _ in range(48)
    ]
    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4,
        foreach=foreach,
    )
    gradients = [torch.randn_like(p) * 0.01 for p in params]
    for p, grad in zip(params, gradients):
        p.grad = grad

    for _ in range(4):
        optimizer.step()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(25):
        optimizer.step()
    torch.xpu.synchronize()
    sec = time.perf_counter() - start
    assert all(torch.isfinite(p).all() for p in params), "optimizer wrote NaN parameters"
    return sec


def main():
    print("Intel XPU:", torch.xpu.get_device_name(0) if torch.xpu.is_available() else "unavailable")
    print("Comparing identical AdamW operations; this does NOT benchmark full RVC epochs")
    eager = measure(False)
    print(f"AdamW foreach=False: {eager:.3f} sec")
    try:
        batched = measure(True)
    except Exception as error:
        print(f"AdamW foreach=True unsupported: {type(error).__name__}: {error}")
        print("Keep APPLIO_XPU_ADAMW_FOREACH=0 (default).")
        return
    print(f"AdamW foreach=True: {batched:.3f} sec")
    if batched < eager * 0.90:
        print("Promising multi-tensor result. Test real epoch with APPLIO_XPU_ADAMW_FOREACH=1.")
    else:
        print("No compelling speed advantage. Keep APPLIO_XPU_ADAMW_FOREACH=0.")
    print("WARNING: foreach uses more temporary GPU memory. Watch iGPU RAM use.")


if __name__ == "__main__":
    main()
