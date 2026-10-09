@echo off
setlocal
cd /d "%~dp0"
echo [Applio XPU] Intel Arc GPU-only installation
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-xpu.ps1"
if errorlevel 1 (
  echo Installation FAILED. Confirm an Intel Arc XPU is available.
  pause
  exit /b 1
)
echo Installation finished.
pause
