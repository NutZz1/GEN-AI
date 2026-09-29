#!/usr/bin/env bash
# ACAI fine-tune of the 128px model + an identical control fine-tune without the critic, then evaluate both.
#   From Windows:  wsl -d Ubuntu -- bash /mnt/c/Users/sreep/OneDrive/Desktop/GEN-AI/run_acai.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/avatarblend-venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoids the memory fragmentation slowdown
COMMON="--size 128 --latent 256 --batch 64 --perc 500 --init checkpoints/vae128.pt --epochs ${EPOCHS:-20} --lr 3e-4"
$PY -u train.py $COMMON --acai "${ACAI:-10000}" --tag _acai | tee results/train128_acai.log
$PY -u train.py $COMMON --acai 0 --tag _ft | tee results/train128_ft.log
for m in vae128_acai.pt vae128_ft.pt vae128.pt; do
  $PY -u evaluate.py --ckpt $m | tee "results/eval_${m%.pt}.log"
done
