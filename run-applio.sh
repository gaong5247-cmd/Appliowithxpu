#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ".venv-xpu/bin/python" ]]; then
  echo "Run ./run-install.sh first (Intel XPU environment missing)" >&2
  exit 1
fi
.venv-xpu/bin/python -c "import torch; assert torch.xpu.is_available(), 'Intel XPU unavailable'; print(torch.xpu.get_device_name(0))"
exec .venv-xpu/bin/python app.py --open "$@"
