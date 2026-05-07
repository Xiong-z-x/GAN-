#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_step() {
  local title="$1"
  shift
  echo
  echo "开始：${title}"
  "$@"
  echo "完成：${title}"
}

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export STYLEGAN3_SEEDS="${STYLEGAN3_SEEDS:-0-7}"
export STYLEGAN3_VIDEO_SEEDS="${STYLEGAN3_VIDEO_SEEDS:-0-7}"
export STYLEGAN3_VIDEO_GRID="${STYLEGAN3_VIDEO_GRID:-2x2}"
export STYLEGAN3_VIDEO_W_FRAMES="${STYLEGAN3_VIDEO_W_FRAMES:-60}"
export STYLEGAN3_VIDEO_OUT="${STYLEGAN3_VIDEO_OUT:-${project_root}/outputs/stylegan3/latent_interpolation.mp4}"
export CYCLEGAN_MODEL_NAME="${CYCLEGAN_MODEL_NAME:-style_vangogh}"
export CYCLEGAN_NUM_TEST="${CYCLEGAN_NUM_TEST:-8}"

run_step "安装 AutoDL 依赖" bash "${project_root}/scripts/install_autodl_dependencies.sh"
run_step "准备外部官方仓库" bash "${project_root}/scripts/setup_external_repos.sh"
run_step "检查外部模型运行条件" python "${project_root}/scripts/check_sota_ready.py" --project-root "${project_root}"
run_step "生成 StyleGAN3 真实人像" bash "${project_root}/scripts/run_stylegan3_generate.sh"
run_step "生成 StyleGAN3 插值视频" bash "${project_root}/scripts/run_stylegan3_video.sh"
run_step "准备动漫化和风格迁移输入" python "${project_root}/scripts/prepare_style_transfer_inputs.py" --clear
run_step "运行 AnimeGANv2 动漫化" bash "${project_root}/scripts/run_animegan2_infer.sh"
run_step "运行 CycleGAN 风格迁移" bash "${project_root}/scripts/run_cyclegan_pretrained_style.sh"

if [ -n "${SG2ADA_TARGET:-}" ]; then
  run_step "运行可选 StyleGAN2-ADA projector" bash "${project_root}/scripts/run_stylegan2ada_projector_optional.sh"
else
  echo
  echo "跳过：未设置 SG2ADA_TARGET，StyleGAN2-ADA projector 保持为可选模块。"
fi

run_step "收集报告素材" python "${project_root}/scripts/collect_report_assets.py"
run_step "检查输出结果" python "${project_root}/scripts/verify_sota_outputs.py" --project-root "${project_root}" --require-video

echo
echo "SOTA 与增强模块流水线执行完成。"
