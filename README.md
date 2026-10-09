# Applio with Intel XPU (experimental)

XPU-only **overlay**, targeting Intel Arc / Core Ultra on Windows. This repository currently stores reproducible patch/build instructions, **not a complete vendored Applio source tree**.

## Build locally (Windows PowerShell)

Install Git and Python, then:

```powershell
git clone https://github.com/IAHispano/Applio.git Applio-XPU
cd Applio-XPU
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Consult https://pytorch.org/get-started/locally/ for the current Intel XPU wheel command.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; assert torch.xpu.is_available(); print(torch.__version__, torch.xpu.get_device_name(0))"
# From a second checkout of this repo, copy scripts/apply_xpu.py to this directory:
python scripts/apply_xpu.py
```

Install Applio's remaining requirements cautiously: default requirements or install launchers may overwrite XPU PyTorch with CUDA/CPU builds. Check `torch.xpu.is_available()` again afterwards. Run `python app.py` only after dependencies are installed.

## Scope and limitations

- Patches device selection, core RVC training autocast and tensors, GPU caching and allocator cleanup.
- Forces single XPU device, with no silent CPU or CUDA fallback in the patched training path.
- **NOT a completed XPU port**: pitch extraction, HuBERT, RMVPE, inference, audio backends, dependency installation, all other device branches and unsupported operators still require audits/tests.
- Intel Meteor Lake Arc graphics uses shared memory; GPU caching can cause memory pressure. Benchmark with caching disabled first.
- BF16 is preferable to FP16 if your installed XPU runtime supports it, but correctness must be validated.
- Do not expect a promised speedup without actual device benchmarks.
- Preserve your existing Applio models/logs/checkpoints in a backup before experimenting.

Upstream: https://github.com/IAHispano/Applio (retain its upstream license and notices).
