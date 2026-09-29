#!/usr/bin/env bash
# Train + evaluate the 128px model on the NVIDIA GPU through WSL2 (Ubuntu).
#   From Windows:  wsl -d Ubuntu -- bash /mnt/c/Users/sreep/OneDrive/Desktop/GEN-AI/run_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/avatarblend-venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible in WSL'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -u train.py --size 128 --epochs "${EPOCHS:-60}" --latent 256 --batch 64 --perc "${PERC:-2000}" | tee results/train128.log
$PY -u evaluate.py | tee results/eval128.log
