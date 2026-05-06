#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${CYCLEGAN_REPO:-${project_root}/external/pytorch-CycleGAN-and-pix2pix}"
input_dir="${CYCLEGAN_INPUT_DIR:-${project_root}/data/processed/style_transfer_inputs}"
out_dir="${CYCLEGAN_RESULTS_DIR:-${project_root}/outputs/cyclegan_style}"
model_name="${CYCLEGAN_MODEL_NAME:-style_vangogh}"
num_test="${CYCLEGAN_NUM_TEST:-50}"

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
bash ./scripts/download_cyclegan_model.sh "${model_name}"

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
