# GAN 真实人像生成大作业

本项目用于完成深度学习课程大作业。核心任务是真实人像生成，增强任务是动漫化与风格迁移展示。项目坚持 GAN 主线，不把 diffusion 模型作为核心方案。

## 1. 项目目标

本项目最终交付内容包括：

- 设计文档：技术路线、模型选择、数据集选择和实验设计。
- 实验代码：手写 DCGAN baseline、SOTA 复现脚本、增强模块脚本。
- 实验结果：生成图、插值视频、动漫化结果、风格迁移结果。
- 实验报告：模型结构说明、结果对比、指标分析、失败案例和伦理说明。

## 2. 技术路线

主线采用“基础模型 + 前沿模型 + 增强展示”的结构：

1. 手写 DCGAN baseline：在 CelebA 64x64 人脸图像上训练，体现课程中 GAN 基础知识。
2. StyleGAN3 官方预训练复现：使用 NVIDIA 官方 FFHQ 预训练权重生成高质量真实人像。
3. AnimeGANv2 动漫化：对个人照片或 StyleGAN3 生成人像做动漫化展示。
4. CycleGAN 风格迁移：使用官方 CycleGAN/pix2pix 仓库与官方风格预训练模型，呼应课堂重点。
5. StyleGAN2-ADA projector：作为可选增强模块，不作为主线成败条件。

关键边界：

- 个人照片不作为 GAN 训练集，只用于 demo、动漫化、风格迁移和可选 projector。
- StyleGAN 系列不从零训练，只做官方预训练权重复现。
- CycleGAN 官方预训练列表没有 face/CelebA/FFHQ 人像域模型，因此人像域 CycleGAN 自训只作为加分项。
- 两天内优先保证可复现实验结果，不把高风险长训练放在主线。

## 3. 推荐 AutoDL 配置

本项目按以下配置规划：

| 项目 | 配置 |
|---|---|
| GPU | RTX 4090D 24GB x1 |
| CUDA | 12.8 |
| PyTorch | 2.8.0 |
| Python | 3.12 镜像，必要时创建 Python 3.10/3.11 环境 |
| 内存 | 60GB |
| 数据盘 | 50GB SSD |

50GB 数据盘不适合下载 FFHQ 1024 全量图像，因此本项目默认依赖官方预训练权重和轻量数据子集。

## 4. AutoDL 最短运行方式

把项目上传到 AutoDL 后，进入项目根目录：

```bash
cd /root/autodl-tmp/GAN
```

运行 SOTA 与增强模块一键流水线：

```bash
bash scripts/run_sota_enhanced_pipeline.sh
```

该命令会自动执行：

1. 安装非 PyTorch 依赖。
2. 克隆外部官方仓库。
3. 检查 SOTA 模块运行条件。
4. 使用 StyleGAN3 官方 FFHQ 预训练权重生成真实人像。
5. 生成 StyleGAN3 潜空间插值视频。
6. 准备 AnimeGANv2 和 CycleGAN 输入。
7. 运行 AnimeGANv2 动漫化。
8. 运行 CycleGAN 官方预训练风格迁移。
9. 收集报告素材。
10. 检查输出结果。

如果 `data/raw/my_photos/` 中有个人照片，增强模块会优先使用个人照片。如果没有个人照片，会使用 StyleGAN3 生成的人像作为输入，保证流程可跑通。

更详细的 AutoDL 说明见：

```text
AutoDL_运行指南.md
```

## 5. 分步运行

准备外部官方仓库：

```bash
bash scripts/setup_external_repos.sh
```

检查 SOTA 模块运行条件：

```bash
python scripts/check_sota_ready.py
```

运行 StyleGAN3 官方预训练生成：

```bash
bash scripts/run_stylegan3_generate.sh
bash scripts/run_stylegan3_video.sh
```

准备个人照片或生成人像作为增强模块输入：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
```

运行 AnimeGANv2 与 CycleGAN：

```bash
bash scripts/run_animegan2_infer.sh
bash scripts/run_cyclegan_pretrained_style.sh
```

收集报告素材：

```bash
python scripts/collect_report_assets.py
```

检查输出结果：

```bash
python scripts/verify_sota_outputs.py
```

## 6. DCGAN Baseline

准备 CelebA 64x64 数据：

```bash
python scripts/prepare_celeba.py \
  --source-dir data/raw/celeba \
  --output-dir data/processed/celeba_64 \
  --image-size 64
```

训练手写 DCGAN：

```bash
python -m src.dcgan.train \
  --data-dir data/processed/celeba_64 \
  --output-dir outputs/dcgan \
  --epochs 25 \
  --batch-size 128
```

快速调试：

```bash
python -m src.dcgan.train \
  --data-dir data/processed/celeba_64 \
  --output-dir outputs/dcgan_debug \
  --epochs 1 \
  --batch-size 32 \
  --max-steps 5
```

## 7. 目录结构

```text
src/dcgan/                         手写 DCGAN baseline
scripts/                           数据处理、SOTA 复现和结果检查脚本
environment/                       环境配置建议
data/raw/                          原始数据占位目录
data/processed/                    处理后数据占位目录
outputs/                           实验输出占位目录
report/                            报告骨架与报告素材目录
external/                          外部官方仓库克隆位置
参考思路/                          本地参考方案材料
```

重要文档：

| 文件 | 作用 |
|---|---|
| `技术路线_最终方案.md` | 最终技术路线与方案论证 |
| `项目完成进度.md` | 项目完成情况和下一步任务 |
| `AutoDL_运行指南.md` | AutoDL 上的直接运行说明 |
| `项目自检报告.md` | 进入 AutoDL 前的质疑式自检记录 |
| `文件说明.md` | 当前各文件和目录的意义 |
| `report/report_outline.md` | 实验报告骨架 |
| `report/results_summary.md` | 实验结果汇总模板 |

## 8. 输出位置

```text
outputs/dcgan/
outputs/stylegan3/images/
outputs/stylegan3/latent_interpolation.mp4
outputs/animegan2/
outputs/cyclegan_style/
outputs/stylegan2ada_projector_optional/
report/report_assets/
```

实际图片、视频、权重、数据集和外部仓库默认不上传 GitHub，只保留目录占位文件。

## 9. 常用可调参数

减少 StyleGAN3 生成数量：

```bash
STYLEGAN3_SEEDS=0-3 bash scripts/run_sota_enhanced_pipeline.sh
```

更换 CycleGAN 官方风格模型：

```bash
CYCLEGAN_MODEL_NAME=style_monet bash scripts/run_sota_enhanced_pipeline.sh
```

更换 AnimeGANv2 权重：

```bash
ANIMEGAN2_STYLE=face_paint_512_v2 bash scripts/run_sota_enhanced_pipeline.sh
```

缩短 StyleGAN3 视频：

```bash
STYLEGAN3_VIDEO_W_FRAMES=30 bash scripts/run_sota_enhanced_pipeline.sh
```

## 10. 本地验证状态

当前已在本地完成以下检查：

```text
python -m compileall src scripts
bash -n scripts/*.sh
python 脚本 --help 检查
DCGAN 前向形状检查
临时图片输入准备检查
临时输出结果检查
```

真实 StyleGAN3、AnimeGANv2、CycleGAN 推理需要在 AutoDL Linux GPU 环境中执行。
