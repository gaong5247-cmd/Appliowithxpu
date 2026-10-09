import torch

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    raise SystemExit("FAILED: Intel XPU unavailable (no CPU fallback)")
print("Device:", torch.xpu.get_device_name(0))
device=torch.device("xpu:0")
a=torch.randn((512,512),device=device,requires_grad=True)
with torch.autocast(device_type="xpu",dtype=torch.bfloat16):
    b=(a@a).square().mean()
b.backward()
torch.xpu.synchronize()
print("XPU matmul + BF16 autocast + backward OK", float(b.detach().cpu()))
