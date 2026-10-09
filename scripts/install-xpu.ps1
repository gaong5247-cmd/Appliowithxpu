# Standalone Windows Intel Arc installer. Never installs CUDA/CPU-only Torch.
# PyTorch 2.14 + XPU: https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

if (-not (Test-Path ".venv-xpu\Scripts\python.exe")) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher (py) not found. Install 64-bit Python 3.12 from https://www.python.org/downloads/"
    }
    $created = $false
    foreach ($version in @("3.12", "3.11")) {
        $valid = $false
        try {
            & py "-$version" -c "import sys; assert sys.maxsize > 2**32" *> $null
            $valid = $LASTEXITCODE -eq 0
        } catch {
            $valid = $false
        }
        if ($valid) {
            Write-Host "Creating Python $version XPU environment..."
            & py "-$version" -m venv ".venv-xpu"
            if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-xpu" }
            $created = $true
            break
        }
    }
    if (-not $created) {
        throw "Install 64-bit Python 3.12 or 3.11 (with py launcher) and rerun."
    }
}
$python = Join-Path $root ".venv-xpu\Scripts\python.exe"
& $python -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32"
if ($LASTEXITCODE -ne 0) { throw "The XPU venv must use 64-bit Python 3.11 or 3.12." }

Write-Host "Installing the exact Intel XPU wheels checked in GitHub Windows CI..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
& $python -m pip install "torch==2.14.1+xpu" "torchaudio==2.11.0+xpu" "torchvision==0.29.1+xpu" --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) { throw "Intel XPU PyTorch installation failed. No CPU/CUDA fallback." }

& $python -c "import torch; assert torch.xpu.is_available(), 'Intel XPU unavailable'; assert torch.__version__.endswith('+xpu'); print('Intel GPU:',torch.xpu.get_device_name(0), 'Torch:',torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "XPU unavailable. Install Intel WHQL driver 32.0.101.8801 or newer, confirm Windows 11 and restart."
}

Write-Host "Installing remaining Applio packages..." -ForegroundColor Cyan
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Applio dependency installation failed." }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Conflicting dependencies detected." }

& $python -c "import torch; assert torch.__version__ == '2.14.1+xpu' and torch.xpu.is_available(); print('XPU PyTorch retained:',torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "A dependency replaced or disabled XPU PyTorch." }

Write-Host "Running actual Intel GPU operator diagnostics..." -ForegroundColor Cyan
& $python scripts/xpu_smoke.py
if ($LASTEXITCODE -ne 0) { throw "XPU operator smoke test failed. See the first exception above." }
Write-Host "Installer complete. Validate the RVC model with run-xpu-check.bat." -ForegroundColor Green
