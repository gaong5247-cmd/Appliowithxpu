@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
  echo Missing XPU environment. Run run-install.bat first.
  pause
  exit /b 1
)
".venv-xpu\Scripts\python.exe" -c "import torch; assert torch.xpu.is_available(), 'Intel XPU unavailable'; print('Starting on', torch.xpu.get_device_name(0))"
if errorlevel 1 (
  echo XPU failed to initialize. CUDA and CPU fallback are disabled.
  pause
  exit /b 1
)
".venv-xpu\Scripts\python.exe" app.py --open %*
if errorlevel 1 pause
