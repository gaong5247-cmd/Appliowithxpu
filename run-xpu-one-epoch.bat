@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
  echo [ERROR] Run run-install.bat first to install Intel XPU PyTorch.
  pause
  exit /b 1
)
echo ====================================================
echo   Intel Arc RVC ONE-EPOCH real training integration
echo ====================================================
echo This generates ONLY temporary synthetic audio and training labels.
echo Existing datasets, experiments and voice checkpoints will NOT be modified.
echo The temporary experiment is removed on success unless --keep is used.
echo.
".venv-xpu\Scripts\python.exe" scripts\xpu_one_epoch.py %*
if errorlevel 1 (
  echo [FAIL] Read the first traceback above. Failed diagnostic logs were kept.
  pause
  exit /b 1
)
echo [PASS] Real RVC XPU training succeeded for one synthetic epoch.
pause
