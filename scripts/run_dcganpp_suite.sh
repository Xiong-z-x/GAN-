#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

run_step() {
  local title="$1"
  shift
  echo
  echo "开始：${title}"
  "$@"
  echo "完成：${title}"
}

last_checkpoint() {
  local pattern="$1"
  ls -1 ${pattern} 2>/dev/null | sort | tail -n 1
}

celeba_root="${DCGAN_CELEBA_ROOT:-data/raw/celeba}"
data_64="${DCGAN_DATA_64:-data/processed/celeba_64}"
data_128_default="${DCGAN_DATA_128_DEFAULT:-data/processed/celeba_128}"
highres_data_dir="${DCGAN_HIGHRES_DATA_DIR:-$data_128_default}"

e0_epochs="${DCGAN_E0_EPOCHS:-60}"
e1_epochs="${DCGAN_E1_EPOCHS:-60}"
e2_epochs="${DCGAN_E2_EPOCHS:-80}"
e3_epochs="${DCGAN_E3_EPOCHS:-80}"
e4_epochs="${DCGAN_E4_EPOCHS:-20}"

if [ ! -d "$data_64" ]; then
  if [ ! -d "$celeba_root" ]; then
    echo "缺少 CelebA 原始数据目录：$celeba_root"
    echo "请先准备数据后再运行。"
    exit 1
  fi
  run_step "准备 CelebA 64" python scripts/prepare_celeba.py --source-dir "$celeba_root" --output-dir "$data_64" --image-size 64
fi

if [ ! -d "$data_128_default" ] && [ "$highres_data_dir" = "$data_128_default" ]; then
  if [ -d "$celeba_root" ]; then
    run_step "准备 CelebA 128" python scripts/prepare_celeba.py --source-dir "$celeba_root" --output-dir "$data_128_default" --image-size 128
  fi
fi

if [ ! -d "$highres_data_dir" ]; then
  echo "缺少高分辨率数据目录：$highres_data_dir"
  echo "如果要走 E3 / E4，请先准备 128 分辨率数据，然后设置 DCGAN_HIGHRES_DATA_DIR。"
  exit 1
fi

run_step "E0 基线 DCGAN" python -m src.dcgan.train \
  --data-dir "$data_64" \
  --output-dir outputs/dcgan_e0_baseline \
  --epochs "$e0_epochs" \
  --batch-size "${DCGAN_E0_BATCH_SIZE:-256}" \
  --image-size 64 \
  --latent-dim "${DCGAN_LATENT_DIM:-100}" \
  --architecture baseline \
  --loss bce

run_step "E1 稳定化 DCGAN" python -m src.dcgan.train \
  --data-dir "$data_64" \
  --output-dir outputs/dcgan_e1_stable \
  --epochs "$e1_epochs" \
  --batch-size "${DCGAN_E1_BATCH_SIZE:-256}" \
  --image-size 64 \
  --latent-dim "${DCGAN_LATENT_DIM:-100}" \
  --architecture baseline \
  --loss hinge \
  --use-spectral-norm \
  --use-ema \
  --dataset-augment \
  --diffaugment color,translation,cutout

run_step "E2 结构增强 DCGAN" python -m src.dcgan.train \
  --data-dir "$data_64" \
  --output-dir outputs/dcgan_e2_residual \
  --epochs "$e2_epochs" \
  --batch-size "${DCGAN_E2_BATCH_SIZE:-192}" \
  --image-size 64 \
  --latent-dim "${DCGAN_LATENT_DIM:-100}" \
  --architecture residual \
  --loss hinge \
  --use-spectral-norm \
  --use-attention \
  --attention-resolutions 32,16 \
  --use-ema \
  --dataset-augment \
  --diffaugment color,translation,cutout

run_step "E3 128 分辨率 DCGAN++" python -m src.dcgan.train \
  --data-dir "$highres_data_dir" \
  --output-dir outputs/dcgan_e3_128 \
  --epochs "$e3_epochs" \
  --batch-size "${DCGAN_E3_BATCH_SIZE:-128}" \
  --image-size 128 \
  --latent-dim "${DCGAN_LATENT_DIM:-100}" \
  --architecture residual \
  --loss hinge \
  --use-spectral-norm \
  --use-attention \
  --attention-resolutions 64,32,16 \
  --use-ema \
  --dataset-augment \
  --diffaugment color,translation,cutout

gen_ckpt="$(last_checkpoint outputs/dcgan_e3_128/checkpoints/generator_epoch_*.pth)"
disc_ckpt="$(last_checkpoint outputs/dcgan_e3_128/checkpoints/discriminator_epoch_*.pth)"

if [ -z "$gen_ckpt" ] || [ -z "$disc_ckpt" ]; then
  echo "未找到 E3 checkpoint，无法继续 E4。"
  exit 1
fi

run_step "E4 R1 细化版" python -m src.dcgan.train \
  --data-dir "$highres_data_dir" \
  --output-dir outputs/dcgan_e4_r1 \
  --epochs "$e4_epochs" \
  --batch-size "${DCGAN_E4_BATCH_SIZE:-128}" \
  --image-size 128 \
  --latent-dim "${DCGAN_LATENT_DIM:-100}" \
  --architecture residual \
  --loss hinge \
  --use-spectral-norm \
  --use-attention \
  --attention-resolutions 64,32,16 \
  --use-ema \
  --use-r1 \
  --r1-gamma "${DCGAN_R1_GAMMA:-10.0}" \
  --r1-interval "${DCGAN_R1_INTERVAL:-16}" \
  --dataset-augment \
  --diffaugment color,translation,cutout \
  --generator-checkpoint "$gen_ckpt" \
  --discriminator-checkpoint "$disc_ckpt"

echo
echo "DCGAN++ 实验序列完成。"
