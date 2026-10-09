@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
  echo [ERROR] Please run run-install.bat to set up Intel XPU first.
  pause
  exit /b 1
)
echo =========================================================
echo    APPLIO INTEL XPU - FULL PHYSICAL DEVICE VALIDATION
echo =========================================================
echo Runs real RVC GPU kernels, G/D forward/backward, RMVPE,
echo HuBERT, synthetic RVC training and audio inference.
echo Saves first traceback and complete log in logs\xpu-diagnostics.
echo No existing user datasets or checkpoints are changed.
echo.
".venv-xpu\Scripts\python.exe" scripts\xpu_diagnostics.py %*
if errorlevel 1 (
  echo [FAIL] See log in logs\xpu-diagnostics.
  pause
  exit /b 1
)
echo [PASS] All requested Intel GPU diagnostic stages passed.
pause
