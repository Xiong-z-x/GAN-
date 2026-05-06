# AutoDL 运行指南

更新时间：2026-05-07

本指南面向后续 Remote-SSH 接入 AutoDL 的继续开发。当前推荐方式是：**本地用 GitHub 同步源码，AutoDL 直接 `git pull`，不要反复压缩传输整个项目**。

## 0. 前提边界

- AutoDL 项目根目录默认：`/root/autodl-tmp/GAN`
- 本地完整结果包可能只存在于本机或迁移盘，不保证 AutoDL 和 GitHub 都有。
- GitHub 仓库只保存源码、文档、轻量交接图，不保存 `data/`、`outputs/`、`external/`、权重和大结果包。
- 个人照片只用于推理展示，不提交公开仓库。
- 不要直接执行外部仓库官方 requirements 覆盖当前 PyTorch 环境，尤其是 InstantID 官方 demo requirements。

## 1. Remote-SSH 登录后同步源码

如果 AutoDL 上没有项目目录：

```bash
cd /root/autodl-tmp
git clone https://github.com/Xiong-z-x/GAN-.git GAN
cd /root/autodl-tmp/GAN
```

如果 AutoDL 上已有项目目录：

```bash
cd /root/autodl-tmp/GAN
git status --short
git pull --ff-only
```

如果 `git pull --ff-only` 因 AutoDL 本地改动失败，先不要强制覆盖，执行：

```bash
git status --short
```

根据输出判断是本地实验产物误入工作区，还是代码文件确实被改动。不要使用 `git reset --hard`，除非确认没有需要保留的本地代码。

## 2. 基础环境准备

```bash
cd /root/autodl-tmp/GAN
source /etc/network_turbo >/dev/null 2>&1 || true
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
pip install -r requirements_autodl.txt
```

说明：

- `requirements_autodl.txt` 只补项目常用依赖，不应主动重装 PyTorch。
- 如果后续需要 InstantID，优先复用已经修好的 AutoDL 环境和本地模型缓存。

## 3. 准备外部仓库

```bash
cd /root/autodl-tmp/GAN
bash scripts/setup_external_repos.sh
python scripts/check_sota_ready.py --project-root /root/autodl-tmp/GAN
```

如果 `check_sota_ready.py` 报某个外部仓库或权重不存在，先确认是否确实需要该模块。`external/` 不提交 GitHub，正常情况下需要在 AutoDL 上重新 clone 或保留旧目录。

## 4. 数据准备

### 4.1 CelebA 64

把 CelebA 原图放到：

```text
data/raw/celeba/
```

执行：

```bash
python scripts/prepare_celeba.py \
  --source-dir data/raw/celeba \
  --output-dir data/processed/celeba_64 \
  --image-size 64
```

### 4.2 FFHQ 128 thumbnails

如果 AutoDL 已经存在：

```text
external/ffhq-dataset/thumbnails128x128/
```

就不要重复下载。

若不存在，优先使用已验证过的 Hugging Face zip 方式或 NVIDIA 官方脚本。下载大数据前先检查磁盘：

```bash
df -h /root/autodl-tmp /root/autodl-fs 2>/dev/null || df -h
```

下载完成后设置：

```bash
export DCGAN_HIGHRES_DATA_DIR=external/ffhq-dataset/thumbnails128x128
```

## 5. DCGAN / DCGAN++ 实验

完整套跑：

```bash
cd /root/autodl-tmp/GAN
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export DCGAN_HIGHRES_DATA_DIR=external/ffhq-dataset/thumbnails128x128
bash scripts/run_dcganpp_suite.sh
```

监控：

```bash
python scripts/monitor_dcganpp_progress.py --project-root /root/autodl-tmp/GAN
```

注意：

- DCGAN 是 baseline，不要期待它接近 StyleGAN3 画质。
- E0/E1/E2/E3/E4 输出目录必须分开，不要把重跑结果覆盖旧结果。
- 如果恢复训练，新输出目录也要单独命名，监控脚本要指向当前目录或确认它读取的是当前实验。

## 6. StyleGAN3 / AnimeGANv2 / CycleGAN

一键增强流程：

```bash
cd /root/autodl-tmp/GAN
bash scripts/run_sota_enhanced_pipeline.sh
```

单独运行 StyleGAN3：

```bash
bash scripts/run_stylegan3_generate.sh
bash scripts/run_stylegan3_video.sh
```

单独运行动漫化：

```bash
ANIMEGAN2_INPUT_DIR=/root/autodl-tmp/GAN/data/processed/style_transfer_inputs_mixed \
ANIMEGAN2_OUTDIR=/root/autodl-tmp/GAN/outputs/animegan2_mix_face_paint_512_v2 \
ANIMEGAN2_STYLE=face_paint_512_v2 \
bash scripts/run_animegan2_infer.sh
```

单独运行 CycleGAN：

```bash
CYCLEGAN_INPUT_DIR=/root/autodl-tmp/GAN/data/processed/style_transfer_inputs_mixed \
CYCLEGAN_RESULTS_DIR=/root/autodl-tmp/GAN/outputs/cyclegan_style_mix_vangogh \
CYCLEGAN_MODEL_NAME=style_vangogh \
CYCLEGAN_NUM_TEST=16 \
bash scripts/run_cyclegan_pretrained_style.sh
```

当前脚本已去掉旧版 `--gpu_ids` 参数，避免与当前 CycleGAN 官方脚本参数不兼容。

## 7. InstantID 本地推理前置检查

InstantID 需要以下文件存在：

```text
external/InstantID/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors
external/InstantID/checkpoints/ip-adapter.bin
external/InstantID/models/antelopev2/
```

基础模型建议本地放在：

```text
/root/autodl-fs/models/YamerMIX_v8
```

或设置：

```bash
export INSTANTID_BASE_MODEL_DIR=/你的/YamerMIX_v8/路径
```

后续运行时必须优先使用本地路径和 `local_files_only=True`，不要让推理脚本临时联网下载 SDXL 大模型。

## 8. FaceGAN Studio

启动 Web 程序：

```bash
cd /root/autodl-tmp/GAN
bash scripts/run_facegan_studio.sh
```

换端口：

```bash
FACEGAN_PORT=7861 bash scripts/run_facegan_studio.sh
```

输出：

```text
outputs/facegan_studio/
report/report_assets/facegan_studio/
```

第一轮验证顺序：

1. 页面能启动。
2. 项目成果展示页能加载 `docs/handoff_assets/` 或已有结果目录。
3. 上传人脸后能生成人脸检测预览。
4. 证件照模块能输出白、蓝、红背景图。
5. AnimeGANv2 模块能生成三种风格图。
6. InstantID 模块能用本地模型生成 4 张图，再扩到 8/16 张。

## 9. 报告素材与报告导出

```bash
python scripts/rebuild_report_figures.py
python scripts/collect_report_assets.py
python scripts/verify_sota_outputs.py --project-root /root/autodl-tmp/GAN --require-video
python scripts/build_final_report_docx.py
```

最终报告：

```text
最终报告/GAN_真实人像生成实验报告_最终版.docx
最终报告/GAN_真实人像生成实验报告_最终版.pdf
```

## 10. 常用检查命令

```bash
git status --short
find outputs -maxdepth 2 -type f | wc -l
find report/report_assets -type f | wc -l
find docs/handoff_assets -type f | wc -l
nvidia-smi
```

## 11. 不要做的事

- 不要把 `GAN_results_images/`、`GAN_new_showcase_results/`、`outputs/`、`data/`、`external/` 整体提交 GitHub。
- 不要把个人照片、训练权重、下载缓存提交 GitHub。
- 不要把 StyleGAN3 跨域 FID 当作公平比较指标。
- 不要把 InstantID 说成 GAN baseline。
- 不要在没有验证的情况下宣称 GFPGAN 已接入完成。
