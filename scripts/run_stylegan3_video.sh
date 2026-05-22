#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${STYLEGAN3_REPO:-${project_root}/external/stylegan3}"
output_file="${STYLEGAN3_VIDEO_OUT:-${project_root}/outputs/stylegan3/latent_interpolation.mp4}"
seeds="${STYLEGAN3_VIDEO_SEEDS:-0-31}"
grid="${STYLEGAN3_VIDEO_GRID:-4x2}"
w_frames="${STYLEGAN3_VIDEO_W_FRAMES:-120}"
truncation="${STYLEGAN3_TRUNC:-0.8}"
network_url="${STYLEGAN3_NETWORK:-https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [ ! -f "${repo_dir}/gen_video.py" ]; then
  echo "未找到 StyleGAN3 仓库，请先运行 scripts/setup_external_repos.sh"
  exit 1
fi

mkdir -p "$(dirname "${output_file}")"
cd "${repo_dir}"
python gen_video.py \
  --output="${output_file}" \
  --trunc="${truncation}" \
  --seeds="${seeds}" \
  --w-frames="${w_frames}" \
  --grid="${grid}" \
  --network="${network_url}"

echo "StyleGAN3 插值视频生成完成：${output_file}"
