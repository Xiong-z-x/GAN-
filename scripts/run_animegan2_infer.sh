#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${ANIMEGAN2_REPO:-${project_root}/external/animegan2-pytorch}"
input_dir="${ANIMEGAN2_INPUT_DIR:-${project_root}/data/processed/style_transfer_inputs}"
out_dir="${ANIMEGAN2_OUTDIR:-${project_root}/outputs/animegan2}"
device_name="${ANIMEGAN2_DEVICE:-cuda:0}"
style_name="${ANIMEGAN2_STYLE:-celeba_distill}"
checkpoint_file="${ANIMEGAN2_CHECKPOINT:-${repo_dir}/weights/${style_name}.pt}"

if [ ! -f "${repo_dir}/test.py" ]; then
  echo "未找到 AnimeGANv2 仓库，请先运行 scripts/setup_external_repos.sh"
  exit 1
fi

if [ ! -f "${checkpoint_file}" ]; then
  echo "未找到 AnimeGANv2 权重文件：${checkpoint_file}"
  echo "请检查仓库 weights 目录，或用 ANIMEGAN2_CHECKPOINT 指定权重。"
  exit 1
fi

if ! find "${input_dir}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | grep -q .; then
  echo "未找到 AnimeGANv2 输入图片：${input_dir}"
  echo "请先运行 scripts/prepare_style_transfer_inputs.py 或放入个人照片。"
  exit 1
fi

mkdir -p "${out_dir}"
cd "${repo_dir}"
python test.py \
  --checkpoint="${checkpoint_file}" \
  --input_dir="${input_dir}" \
  --output_dir="${out_dir}" \
  --device="${device_name}" \
  --x32

echo "AnimeGANv2 动漫化推理完成：${out_dir}"
