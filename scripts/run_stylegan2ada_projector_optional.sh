#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${SG2ADA_REPO:-${project_root}/external/stylegan2-ada-pytorch}"
target_image="${SG2ADA_TARGET:-}"
out_dir="${SG2ADA_OUTDIR:-${project_root}/outputs/stylegan2ada_projector_optional}"
network_url="${SG2ADA_NETWORK:-https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl}"
num_steps="${SG2ADA_NUM_STEPS:-1000}"

if [ -z "${target_image}" ]; then
  echo "请用 SG2ADA_TARGET 指定一张已对齐的人脸图像。"
  exit 1
fi

if [ ! -f "${repo_dir}/projector.py" ]; then
  echo "未找到 StyleGAN2-ADA 仓库，请先运行 scripts/setup_external_repos.sh"
  exit 1
fi

mkdir -p "${out_dir}"
cd "${repo_dir}"
python projector.py \
  --outdir="${out_dir}" \
  --target="${target_image}" \
  --num-steps="${num_steps}" \
  --network="${network_url}"

echo "StyleGAN2-ADA projector 完成：${out_dir}"
