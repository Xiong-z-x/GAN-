#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${STYLEGAN3_REPO:-${project_root}/external/stylegan3}"
out_dir="${STYLEGAN3_OUTDIR:-${project_root}/outputs/stylegan3/images}"
seeds="${STYLEGAN3_SEEDS:-0-31}"
truncation="${STYLEGAN3_TRUNC:-0.8}"
noise_mode="${STYLEGAN3_NOISE_MODE:-const}"
network_url="${STYLEGAN3_NETWORK:-https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [ ! -f "${repo_dir}/gen_images.py" ]; then
  echo "未找到 StyleGAN3 仓库，请先运行 scripts/setup_external_repos.sh"
  exit 1
fi

mkdir -p "${out_dir}"
cd "${repo_dir}"
python gen_images.py \
  --outdir="${out_dir}" \
  --trunc="${truncation}" \
  --noise-mode="${noise_mode}" \
  --seeds="${seeds}" \
  --network="${network_url}"

echo "StyleGAN3 人像生成完成：${out_dir}"
