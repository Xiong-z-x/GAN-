#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="${project_root}/external"
mkdir -p "${external_dir}"

if ! command -v git >/dev/null 2>&1; then
  echo "未找到 git，请先在 AutoDL 环境中安装 git。"
  exit 1
fi

clone_repo() {
  local_dir="$1"
  repo_url="$2"
  if [ -d "${external_dir}/${local_dir}/.git" ]; then
    echo "已存在：${local_dir}"
  else
    git clone --depth 1 "${repo_url}" "${external_dir}/${local_dir}"
  fi
}

clone_repo "stylegan3" "https://github.com/NVlabs/stylegan3.git"
clone_repo "pytorch-CycleGAN-and-pix2pix" "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git"
clone_repo "animegan2-pytorch" "https://github.com/bryandlee/animegan2-pytorch.git"
clone_repo "stylegan2-ada-pytorch" "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
clone_repo "contrastive-unpaired-translation" "https://github.com/taesungp/contrastive-unpaired-translation.git"

echo "外部仓库准备完成。"
