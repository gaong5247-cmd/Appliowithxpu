@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
  echo [ERROR] XPU environment not installed. Run run-install.bat.
  pause
  exit /b 1
)
echo ===============================================
echo   Applio Intel XPU end-to-end GPU diagnostics
echo ===============================================
".venv-xpu\Scripts\python.exe" scripts\xpu_smoke.py
if errorlevel 1 goto error
".venv-xpu\Scripts\python.exe" scripts\test_slices.py
if errorlevel 1 goto error
".venv-xpu\Scripts\python.exe" scripts\xpu_model_step.py
if errorlevel 1 goto error
echo.
echo [PASS] XPU operations and synthetic RVC training step passed.
echo Real dataset training, pitch extraction, and realtime audio need separate testing.
pause
exit /b 0
:error
echo.
echo [FAILED] Intel XPU test failed. Copy the traceback above for diagnosis.
pause
exit /b 1
