#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${CYCLEGAN_REPO:-${project_root}/external/pytorch-CycleGAN-and-pix2pix}"
input_dir="${CYCLEGAN_INPUT_DIR:-${project_root}/data/processed/style_transfer_inputs}"
out_dir="${CYCLEGAN_RESULTS_DIR:-${project_root}/outputs/cyclegan_style}"
model_name="${CYCLEGAN_MODEL_NAME:-style_vangogh}"
num_test="${CYCLEGAN_NUM_TEST:-50}"
checkpoint_file="${repo_dir}/checkpoints/${model_name}_pretrained/latest_net_G.pth"
min_checkpoint_bytes="${CYCLEGAN_MIN_CHECKPOINT_BYTES:-40000000}"

if [ ! -f "${repo_dir}/test.py" ]; then
  echo "未找到 CycleGAN 仓库，请先运行 scripts/setup_external_repos.sh"
  exit 1
fi

if ! find "${input_dir}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | grep -q .; then
  echo "未找到 CycleGAN 输入图片：${input_dir}"
  echo "请先运行 scripts/prepare_style_transfer_inputs.py 或放入个人照片。"
  exit 1
fi

cd "${repo_dir}"
checkpoint_bytes=0
if [ -f "${checkpoint_file}" ]; then
  checkpoint_bytes="$(stat -c%s "${checkpoint_file}")"
fi

if [ "${checkpoint_bytes}" -ge "${min_checkpoint_bytes}" ]; then
  echo "使用已有 CycleGAN 权重：${checkpoint_file} (${checkpoint_bytes} bytes)"
else
  echo "CycleGAN 权重缺失或不完整，将下载到临时文件后再替换：${checkpoint_file}"
  mkdir -p "$(dirname "${checkpoint_file}")"
  tmp_checkpoint="${checkpoint_file}.download"
  rm -f "${tmp_checkpoint}"
  wget \
    "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/${model_name}.pth" \
    -O "${tmp_checkpoint}"
  downloaded_bytes="$(stat -c%s "${tmp_checkpoint}")"
  if [ "${downloaded_bytes}" -lt "${min_checkpoint_bytes}" ]; then
    echo "下载后的 CycleGAN 权重大小异常：${tmp_checkpoint} (${downloaded_bytes} bytes)"
    exit 1
  fi
  mv "${tmp_checkpoint}" "${checkpoint_file}"
fi

python test.py \
  --dataroot "${input_dir}" \
  --name "${model_name}_pretrained" \
  --model test \
  --no_dropout \
  --num_test "${num_test}" \
  --results_dir "${out_dir}" \
  --preprocess resize \
  --load_size 256

echo "CycleGAN 风格迁移完成：${out_dir}"
