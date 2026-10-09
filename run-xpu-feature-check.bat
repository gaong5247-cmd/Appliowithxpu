@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
  echo [ERROR] Please run run-install.bat before testing Intel XPU.
  pause
  exit /b 1
)
echo =============================================================
echo   Intel Arc RMVPE and HuBERT real GPU feature extraction
echo =============================================================
echo Uses eight generated synthetic audio clips in a unique temp folder.
echo May need to download Applio pretrained models with --download.
echo No personal datasets are accessed or changed.
echo.
".venv-xpu\Scripts\python.exe" scripts\xpu_feature_check.py %*
if errorlevel 1 (
  echo [FAIL] Intel Arc feature extraction. Read the first traceback above.
  pause
  exit /b 1
)
echo [PASS] XPU pitch and speech embeddings produced successfully.
pause
