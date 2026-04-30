# 调研发现

更新时间：2026-04-30

## 1. 本地参考材料

- `参考思路/GPT参考思路：deep-research-report.md`：主张 DCGAN + StyleGAN2-ADA + CycleGAN，是最终方案主体。
- `参考思路/gemini参考思路：GAN 真实人像生成大作业方案 - Google Gemini.pdf`：主张 CycleGAN/CUT 做非成对域转换，适合作为 CycleGAN 支线与备用思路。

## 2. 课程范围

李宏毅 ML 2021 课程页在 Generative Model 中列出 GAN Basic、Theory of GAN and WGAN、Evaluation of GAN and Conditional GAN、CycleGAN，并有 HW6 GAN。  
来源：https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php

## 3. 仓库与权重

- StyleGAN2-ADA 官方仓库存在，提供 FFHQ、MetFaces 等预训练 pkl，并提供 `generate.py`、`style_mixing.py`、`projector.py`、`calc_metrics.py`。
  - 来源：https://github.com/NVlabs/stylegan2-ada-pytorch
- StyleGAN3 官方仓库存在，提供 FFHQ/MetFaces 等预训练 pkl、`gen_images.py`、`gen_video.py`、`calc_metrics.py`，但自拍 projector 不如 StyleGAN2-ADA 主线直接。
  - 来源：https://github.com/NVlabs/stylegan3
- CycleGAN/pix2pix 官方 PyTorch 仓库存在，2025 README 提到支持 Python 3.11 与 PyTorch 2.4；但官方预训练列表没有 face-domain 模型。
  - 仓库：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  - 预训练列表脚本：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh
- CUT/FastCUT 官方仓库存在，说明相比 CycleGAN 更快、更省显存；但它是图像翻译备选，不是本项目真实人像生成主线。
  - 来源：https://github.com/taesungp/contrastive-unpaired-translation

## 4. 数据集

- CelebA 官方存在，标准人脸数据集，适合 DCGAN baseline。
  - 来源：https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- FFHQ 官方存在，70,000 张 1024x1024 高质量人脸 PNG，并提供 128x128 thumbnails；是 StyleGAN 人脸标准域。
  - 来源：https://github.com/NVlabs/ffhq-dataset
- MetFaces 官方存在，1336 张 1024x1024 艺术肖像人脸，适合 CycleGAN 附录。
  - 来源：https://github.com/NVlabs/metfaces-dataset

## 5. 负面发现

- 未找到一个官方仓库能同时满足全部需求。
- 未找到官方人像域 CycleGAN 预训练权重。
- 近 2-3 年更强的人像生成多由 diffusion 主导；本作业强制 GAN-centered，因此不应把 diffusion 作为主线。
- 自己照片数量不足以训练 GAN，只适合 demo/projector。

## 6. 新增约束与修正

- 用户明确修正：主任务仍是真实人像生成，但可以加入动漫化/风格迁移模块。
- 用户提供 AutoDL 截图：PyTorch 2.8.0、Python 3.12、Ubuntu 22.04、CUDA 12.8、RTX 4090D 24GB、16 vCPU、60GB 内存、数据盘 50GB SSD。
- 工程影响：50GB 数据盘不适合下载 FFHQ 1024 全量；应依赖官方预训练权重和轻量数据。
- 工程影响：StyleGAN2-ADA 官方环境偏旧，projector 降为可选；StyleGAN3 官方预训练生成更适合当前高配环境主线。
- 动漫化增强：AnimeGANv2 PyTorch 仓库可作为预训练推理模块；已核对 `test.py` 参数，仓库权重目录包含 `celeba_distill.pt`。来源 https://github.com/bryandlee/animegan2-pytorch
- 风格迁移增强：CycleGAN 官方预训练风格模型可作为零训练课程展示，来源 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh
