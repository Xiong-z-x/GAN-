# GAN 真实人像生成课程项目

本项目用于深度学习课程大作业，主线是**基于 GAN 的真实人像生成**，扩展为**动漫化、艺术风格迁移和输入人脸身份保持应用**。当前仓库已进入“迁移前封板”状态：源码、文档和少量代表性交接素材适合提交 GitHub；大数据集、外部仓库、模型权重和完整结果包不直接提交。

## 当前总路线

| 层级 | 模块 | 定位 | 当前状态 |
|---|---|---|---|
| 课程 baseline | DCGAN / DCGAN++ | 手写 GAN 训练、展示基础生成能力和改进尝试 | 代码已实现，AutoDL 已跑过 E0/E1，E2 有中断/续训历史 |
| 高质量 GAN 上限 | StyleGAN3 | 使用 NVIDIA 官方 FFHQ 预训练权重展示成熟 GAN 质量上限 | 已生成代表性人像与交接展示图 |
| GAN 风格迁移 | AnimeGANv2 / CycleGAN | 动漫风格、梵高/莫奈/浮世绘等风格迁移 | 已跑过多权重/多风格结果，脚本已兼容当前 CycleGAN 参数 |
| 身份保持应用 | InstantID + 可选 GFPGAN | 输入个人人脸，生成不同姿态、造型和场景；GFPGAN 后续作为修复增强 | InstantID 已有展示成果；GFPGAN 仍是待验证增强 |
| 应用封装 | FaceGAN Studio | Gradio Web 程序，统一入口展示和调用上述能力 | 第一版代码已加入，需在 AutoDL 环境继续验证重模型功能 |

严格边界：

- DCGAN / DCGAN++ 是课程基础 GAN 展示，不应被包装成接近 StyleGAN3 的高质量人脸生成器。
- StyleGAN3 使用官方 FFHQ 预训练权重，是高质量 GAN 上限展示，不是本项目从零训练成果。
- AnimeGANv2 / CycleGAN 是风格迁移增强模块，不参与 DCGAN FID 主指标排序。
- InstantID 属于身份保持 diffusion 应用扩展，用于“我的脸 + 不同姿态/造型”，不是 GAN baseline。

## 迁移与接手入口

后续推荐使用 VS Code / Cursor 的 Remote-SSH 直接进入 AutoDL。源码同步优先走 GitHub，而不是反复打包压缩。

AutoDL 项目目录约定：

```text
/root/autodl-tmp/GAN
```

新会话或新接手模型必须优先阅读：

1. `docs/handoff/新对话初始化提示词.md`
2. `docs/handoff/迁移前交接总报告.md`
3. `docs/handoff/后续模型注意事项.md`
4. `AutoDL_运行指南.md`
5. `FaceGAN_Studio_运行说明.md`
6. `代码附录.md`

## 目录结构

| 路径 | 作用 | GitHub 策略 |
|---|---|---|
| `src/dcgan/` | 手写 DCGAN / DCGAN++ 训练代码 | 提交 |
| `scripts/` | 数据准备、模型调用、报告整理、监控脚本 | 提交 |
| `facegan_studio/` | Gradio Web 应用封装 | 提交 |
| `docs/superpowers/` | FaceGAN Studio 设计与实施计划 | 提交 |
| `docs/handoff/` | 迁移交接、经验、下个模型初始化提示词 | 提交 |
| `docs/handoff_assets/` | 少量代表性展示图，便于 AutoDL 从 GitHub 直接看到成果 | 提交 |
| `environment/` | 环境配置建议 | 提交 |
| `最终报告/` | Word/PDF 最终报告 | 提交 |
| `data/` | 原始数据和中间数据 | 不提交 |
| `outputs/` | 训练和推理结果 | 不提交 |
| `report/report_assets/` | 大量报告素材 | 不提交 |
| `external/` | 第三方官方仓库 | 不提交 |
| `GAN_results_images/` | 完整结果包 | 不提交 |
| `GAN_new_showcase_results/` | 新展示结果包 | 不提交 |

## AutoDL 快速启动

在 AutoDL 上从 GitHub 拉取或更新源码：

```bash
cd /root/autodl-tmp
git clone https://github.com/Xiong-z-x/GAN-.git GAN
cd /root/autodl-tmp/GAN
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

如果 AutoDL 上已经有项目目录：

```bash
cd /root/autodl-tmp/GAN
git status --short
git pull --ff-only
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

安装依赖时不要随意重装 PyTorch：

```bash
pip install -r requirements_autodl.txt
```

完整运行细节见 `AutoDL_运行指南.md`。

## FaceGAN Studio

启动：

```bash
cd /root/autodl-tmp/GAN
bash scripts/run_facegan_studio.sh
```

默认端口：`7860`。

输出目录：

```text
outputs/facegan_studio/
report/report_assets/facegan_studio/
```

InstantID 模块需要本地已有：

```text
external/InstantID/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors
external/InstantID/checkpoints/ip-adapter.bin
external/InstantID/models/antelopev2/
```

并且需要 SDXL 基础模型目录。程序会按顺序查找：

```text
INSTANTID_BASE_MODEL_DIR
/root/autodl-fs/models/YamerMIX_v8
/autodl-fs/data/models/YamerMIX_v8
/root/autodl-fs/data/models/YamerMIX_v8
```

## 关键结果边界

- DCGAN 旧主结果：CelebA_64 训练 60 epoch，10k 生成图 FID 记录为 `23.1820`。
- StyleGAN3 跨域 FID 曾记录为 `216.2482`，但该结果是 FFHQ 生成图对 CelebA_64 真实集的跨域参考，不能用于说明 StyleGAN3 质量差。
- 新展示结果目录 `GAN_new_showcase_results/` 约 300MB，不提交 GitHub；其中 5 张代表图已复制到 `docs/handoff_assets/`。
- 个人照片只用于推理展示，不进入训练集，不应上传到公开仓库。

## 主要文档

| 文件 | 用途 |
|---|---|
| `技术路线_最终方案.md` | 早期完整技术路线和事实边界 |
| `GAN_真实人像生成实验报告.md` | 课程报告正文底稿 |
| `代码附录.md` | 自写代码、脚本和模块说明 |
| `报告素材使用说明.md` | 报告图片和素材用法 |
| `最终提交清单.md` | 课程提交内容边界 |
| `FaceGAN_Studio_运行说明.md` | Web 程序运行说明 |
| `docs/handoff/迁移前交接总报告.md` | 当前封板总状态 |
| `docs/handoff/后续模型注意事项.md` | 最容易出错的历史经验 |
| `docs/handoff/新对话初始化提示词.md` | 下一个新会话可直接复制使用的初始化提示词 |

## 后续改进顺序

1. 在 AutoDL Remote-SSH 环境确认 GitHub 拉取、依赖、外部仓库和本地模型路径。
2. 启动 FaceGAN Studio，先验证不依赖重模型的页面、成果展示和证件照模块。
3. 验证 AnimeGANv2 / CycleGAN 调用链，确保输出进入 `outputs/facegan_studio/`。
4. 验证 InstantID 本地模型加载与 4/8 张姿态生成，必要时降低分辨率或步数。
5. 再考虑接入 GFPGAN，先作为可选后处理，不要破坏已跑通的主流程。
6. 最后更新实验报告，新增 FaceGAN Studio 应用封装章节和新结果图。
