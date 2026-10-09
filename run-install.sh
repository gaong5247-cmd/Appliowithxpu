#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v python3.12 >/dev/null || { echo "Python 3.12 required" >&2; exit 1; }
python3.12 -m venv .venv-xpu
.venv-xpu/bin/python -m pip install --upgrade pip setuptools wheel
.venv-xpu/bin/python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
.venv-xpu/bin/python -c "import torch; assert torch.xpu.is_available(), 'Intel XPU unavailable'"
.venv-xpu/bin/python -m pip install -r requirements.txt
.venv-xpu/bin/python scripts/xpu_smoke.py
echo "Ready: ./run-applio.sh"
