# Standalone Windows installer. No CUDA/CPU PyTorch fallback.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

if (!(Test-Path ".venv-xpu\Scripts\python.exe")) {
    $created = $false
    foreach ($version in @("3.12", "3.11")) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$version" -c "import sys; print(sys.version)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py "-$version" -m venv ".venv-xpu"
                if ($LASTEXITCODE -ne 0) { throw "Failed to create Python venv" }
                $created = $true
                break
            }
        }
    }
    if (!$created) {
        throw "Install Python 3.12 from https://www.python.org/downloads/ (including the py launcher), then rerun."
    }
}
$python = Join-Path $root ".venv-xpu\Scripts\python.exe"
Write-Host "Installing PyTorch XPU for Intel Arc..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
& $python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) { throw "XPU PyTorch wheel installation failed. No fallback permitted." }
& $python -c "import torch; assert torch.xpu.is_available(), 'Intel XPU unavailable'; print('XPU:',torch.xpu.get_device_name(0),torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "XPU unavailable. Update Intel Arc Graphics drivers and check Windows 11." }

Write-Host "Installing Applio dependencies (preserving XPU PyTorch)..." -ForegroundColor Cyan
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $python -c "import torch; assert torch.xpu.is_available(); print('Confirmed',torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Dependencies broke PyTorch XPU" }
& $python scripts/xpu_smoke.py
if ($LASTEXITCODE -ne 0) { throw "XPU smoke test failed" }
Write-Host "XPU environment ready. Launch with run-applio.bat" -ForegroundColor Green
