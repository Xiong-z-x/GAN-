#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "开始检查 Python 与 PyTorch 环境。"
python - <<'PY'
import sys

print(f"Python: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    raise SystemExit("Python 版本过低，建议使用 Python 3.10 或 3.11。")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"未检测到 PyTorch，请先使用 AutoDL PyTorch 镜像或安装 PyTorch：{exc}") from exc

print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "开始安装项目所需的非 PyTorch 依赖。"
python -m pip install --upgrade pip
python -m pip install -r "${project_root}/requirements_autodl.txt"

echo "依赖安装完成。"
