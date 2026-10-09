"""Apply an XPU-only overlay to a clean IAHispano/Applio checkout.

This is an experimental port, not a claim of end-to-end RVC compatibility.
Fail closed whenever upstream source changes unexpectedly.
"""
from pathlib import Path
import sys

def change(path, old, new, count=1):
    p=Path(path)
    s=p.read_text(encoding="utf-8")
    actual=s.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} occurrence(s), found {actual}: {old[:90]!r}")
    p.write_text(s.replace(old,new),encoding="utf-8")
    print(f"patched {path}: {old[:55]!r}")

def main():
    if not Path("rvc/train/train.py").exists():
        sys.exit("Run this script from the upstream Applio checkout root")
    path="rvc/configs/config.py"
    change(path,'self.device = "cuda:0" if torch.cuda.is_available() else "cpu"',
           'self.device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else (_ for _ in ()).throw(RuntimeError("Intel XPU is required; install an XPU-enabled PyTorch build"))')
    change(path,'if self.device.startswith("cuda")\n            else None','if self.device.startswith("cuda")\n            else torch.xpu.get_device_name(0)')
    change(path,'if self.device.startswith("cuda"):\n            self.set_cuda_config()\n        else:\n            self.device = "cpu"',
           'if self.device.startswith("xpu"):\n            self.gpu_name = torch.xpu.get_device_name(0)\n            self.gpu_mem = int(torch.xpu.get_device_properties(0).total_memory // (1024**3))')
    start=Path(path).read_text(encoding="utf-8")
    start=start[:start.index("def max_vram_gpu(gpu):")] + '''def max_vram_gpu(gpu):
    return round(torch.xpu.get_device_properties(int(gpu)).total_memory / (1024**3))

def get_gpu_info():
    if not torch.xpu.is_available():
        return "Intel XPU not available"
    return "\\n".join(
        f"{i}: {torch.xpu.get_device_name(i)} ({max_vram_gpu(i)} GB)"
        for i in range(torch.xpu.device_count())
    )

def get_number_of_gpus():
    return "-".join(str(i) for i in range(torch.xpu.device_count())) if torch.xpu.is_available() else "-"
'''
    Path(path).write_text(start,encoding="utf-8")

    path="rvc/train/train.py"
    change(path,'and torch.cuda.is_available()\n            and torch.cuda.is_bf16_supported()',
           'and hasattr(torch, "xpu") and torch.xpu.is_available()')
    change(path,'elif precision == "fp16" and torch.cuda.is_available():',
           'elif precision == "fp16" and hasattr(torch, "xpu") and torch.xpu.is_available():')
    begin=Path(path).read_text(encoding="utf-8")
    a=begin.index('    if torch.cuda.is_available():\n        device = torch.device("cuda")')
    b=begin.index('\n    def start():',a)
    begin=begin[:a]+'''    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is mandatory; refusing CPU/CUDA fallback")
    device = torch.device("xpu:0")
    gpus = [0]
    n_gpus = 1
    print("Applio XPU experimental single-device training:", torch.xpu.get_device_name(0))
''' +begin[b:]
    Path(path).write_text(begin,encoding="utf-8")
    change(path,'if torch.cuda.is_available():\n        torch.cuda.set_device(device_id)',
           'if device.type == "xpu":\n        torch.xpu.set_device(device_id)')
    change(path,'if torch.cuda.is_available():\n        net_g = net_g.cuda(device_id)\n        net_d = net_d.cuda(device_id)\n    else:\n        net_g = net_g.to(device)\n        net_d = net_d.to(device)',
           'net_g = net_g.to(device)\n    net_d = net_d.to(device)')
    change(path,'use_scaler = device.type == "cuda" and train_dtype == torch.float16',
           'use_scaler = device.type == "xpu" and train_dtype == torch.float16')
    change(path,'scaler = torch.amp.GradScaler(enabled=use_scaler)',
           'scaler = torch.amp.GradScaler("xpu", enabled=use_scaler)')
    change(path,'use_amp = device.type == "cuda" and (',
           'use_amp = device.type == "xpu" and (')
    change(path,'if device.type == "cuda" and cache_data_in_gpu:',
           'if device.type == "xpu" and cache_data_in_gpu:')
    change(path,'if device.type == "cuda" and not cache_data_in_gpu:',
           'if device.type == "xpu" and not cache_data_in_gpu:')
    change(path,'elif device.type != "cuda":',
           'elif device.type != "xpu":')
    change(path,'[tensor.cuda(device_id, non_blocking=True) for tensor in info]',
           '[tensor.to(device, non_blocking=True) for tensor in info]',2)
    change(path,'device_type="cuda", enabled=use_amp, dtype=train_dtype',
           'device_type="xpu", enabled=use_amp, dtype=train_dtype',4)
    change(path,'torch.cuda.empty_cache()','torch.xpu.empty_cache()',2)
    print("XPU overlay applied. Full training/inference compatibility is NOT yet verified.")

if __name__=="__main__":
    main()
