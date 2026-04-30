# AutoDL 运行指南

更新时间：2026-04-30

本文档用于把本项目迁移到 AutoDL 后快速运行。默认你使用 PyTorch 镜像，且 GPU、CUDA 和磁盘配置满足项目要求。

## 1. 最短运行路径

进入项目根目录：

```bash
cd /root/autodl-tmp/GAN
```

运行 SOTA 与增强模块流水线：

```bash
bash scripts/run_sota_enhanced_pipeline.sh
```

这条命令会依次执行：

1. 安装非 PyTorch 依赖。
2. 克隆外部官方仓库。
3. 检查 SOTA 模块运行条件。
4. 使用 StyleGAN3 官方 FFHQ 预训练权重生成真实人像。
5. 生成 StyleGAN3 潜空间插值视频。
6. 准备动漫化和风格迁移输入。
7. 运行 AnimeGANv2 动漫化。
8. 运行 CycleGAN 官方预训练风格迁移。
9. 收集报告素材。
10. 检查输出结果。

如果 `data/raw/my_photos/` 中有个人照片，流水线会优先使用个人照片作为动漫化和风格迁移输入。如果没有个人照片，流水线会使用 StyleGAN3 生成的人像作为输入，保证增强模块仍能跑通。

## 2. 个人照片使用方式

把照片放入：

```text
data/raw/my_photos/
```

支持格式：

```text
jpg, jpeg, png, bmp, webp
```

然后重新运行：

```bash
bash scripts/run_sota_enhanced_pipeline.sh
```

个人照片只用于展示，不用于训练 GAN。

## 3. 常用可调参数

减少 StyleGAN3 生成数量：

```bash
STYLEGAN3_SEEDS=0-3 bash scripts/run_sota_enhanced_pipeline.sh
```

更换 CycleGAN 官方风格模型：

```bash
CYCLEGAN_MODEL_NAME=style_monet bash scripts/run_sota_enhanced_pipeline.sh
```

可选模型名包括：

```text
style_monet, style_cezanne, style_ukiyoe, style_vangogh
```

更换 AnimeGANv2 权重：

```bash
ANIMEGAN2_STYLE=face_paint_512_v2 bash scripts/run_sota_enhanced_pipeline.sh
```

可选权重包括：

```text
celeba_distill, face_paint_512_v1, face_paint_512_v2, paprika
```

缩短 StyleGAN3 视频：

```bash
STYLEGAN3_VIDEO_W_FRAMES=30 bash scripts/run_sota_enhanced_pipeline.sh
```

## 4. 单独运行 SOTA 模块

只准备外部仓库：

```bash
bash scripts/setup_external_repos.sh
```

只检查运行条件：

```bash
python scripts/check_sota_ready.py
```

只生成 StyleGAN3 图片：

```bash
bash scripts/run_stylegan3_generate.sh
```

只运行 AnimeGANv2：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
bash scripts/run_animegan2_infer.sh
```

只运行 CycleGAN：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
bash scripts/run_cyclegan_pretrained_style.sh
```

## 5. 手写 DCGAN baseline

准备 CelebA 64x64：

```bash
python scripts/prepare_celeba.py \
  --source-dir data/raw/celeba \
  --output-dir data/processed/celeba_64 \
  --image-size 64
```

训练 baseline：

```bash
python -m src.dcgan.train \
  --data-dir data/processed/celeba_64 \
  --output-dir outputs/dcgan \
  --epochs 25 \
  --batch-size 128
```

## 6. 输出位置

```text
outputs/stylegan3/images/
outputs/stylegan3/latent_interpolation.mp4
outputs/animegan2/
outputs/cyclegan_style/
report/report_assets/
```

实验报告优先使用 `report/report_assets/` 中的素材。

## 7. 常见问题

如果提示未找到 `nvcc`，说明 StyleGAN3 自定义算子可能无法编译。优先切换到带完整 CUDA toolkit 的 AutoDL PyTorch 镜像。

如果 CycleGAN 下载权重失败，通常是网络访问问题。重新运行同一条命令即可，脚本会复用已有目录。

如果 AnimeGANv2 提示没有输入图片，先运行：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
```

如果报告素材为空，先确认至少运行过 StyleGAN3、AnimeGANv2 或 CycleGAN 中的一个模块，再执行：

```bash
python scripts/collect_report_assets.py
```
