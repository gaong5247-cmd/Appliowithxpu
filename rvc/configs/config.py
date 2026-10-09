"""XPU-only device configuration for Applio.

GPU compute requires Intel Arc / Intel XPU. Host-side audio processing,
FAISS indexes and checkpoint deserialization can still run on the CPU.
"""
import json
import os

import torch

version_config_paths = ["48000.json", "40000.json", "32000.json", "24000.json"]


def require_xpu(index=0):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "Applio with XPU requires a working Intel XPU GPU. "
            "Update your Intel graphics driver, install PyTorch from "
            "https://download.pytorch.org/whl/xpu and run scripts/xpu_smoke.py. "
            "CPU/CUDA fallback has deliberately been disabled."
        )
    if index < 0 or index >= torch.xpu.device_count():
        raise RuntimeError(f"XPU device {index} is not available")
    return torch.device(f"xpu:{index}")


def max_vram_gpu(gpu):
    """Reported XPU memory in GiB (integrated GPU memory can be shared)."""
    props = torch.xpu.get_device_properties(int(gpu)) if torch.xpu.is_available() else None
    return round(props.total_memory / (1024**3)) if props is not None else 0


def get_gpu_info():
    try:
        require_xpu()
    except RuntimeError as exc:
        return f"Intel XPU unavailable: {exc}"
    return "\n".join(
        f"{i}: {torch.xpu.get_device_name(i)} ({max_vram_gpu(i)} GiB reported)"
        for i in range(torch.xpu.device_count())
    )


def get_number_of_gpus():
    # Applio XPU training currently uses a single GPU, not XCCL DDP.
    return "0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "-"


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = str(require_xpu())
        self.gpu_name = torch.xpu.get_device_name(0)
        self.gpu_mem = max_vram_gpu(0)
        self.json_config = self.load_config_json()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
        # XPU training starts single-device: model compute on XPU, audio IO on host.
        print(f"[Applio XPU] {self.gpu_name}: {self.device}; reported memory {self.gpu_mem} GiB")

    def load_config_json(self):
        configs = {}
        for name in version_config_paths:
            path = os.path.join("rvc", "configs", name)
            with open(path, "r", encoding="utf-8") as f:
                configs[name] = json.load(f)
        return configs

    def device_config(self):
        # Conservative defaults for memory-constrained integrated Arc devices.
        # Intel Arc iGPU reports shared system RAM, not dedicated VRAM.
        props = torch.xpu.get_device_properties(0)
        if getattr(props, "is_integrated_gpu", False) or (self.gpu_mem and self.gpu_mem <= 4):
            return (1, 5, 30, 32)
        return (1, 6, 38, 41)
