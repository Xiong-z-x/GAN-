# 基于 GAN 的真实人像生成与风格迁移扩展实验报告

## 摘要

本项目围绕深度学习课程中的生成对抗网络（Generative Adversarial Network, GAN）展开，核心任务是真实人像生成，扩展任务为人像动漫化与艺术风格迁移。为了体现从基础理论到工程复现的学习过程，实验采用循序渐进的技术路线：首先手写并训练 DCGAN baseline，在标准公开数据集 CelebA 上验证基础 GAN 的生成能力；随后复现 NVIDIA StyleGAN3 官方 FFHQ 预训练模型，展示现代 style-based GAN 在高质量真实人像生成上的优势；最后使用 AnimeGANv2 与 CycleGAN 对个人照片和生成头像进行动漫化、浮世绘、莫奈和梵高风格迁移，呼应课程中 CycleGAN 与非成对图像翻译的重点内容。

实验在 AutoDL GPU 环境中完成。DCGAN 使用 CelebA Align&Cropped 数据集处理得到的 `CelebA_64` 训练 60 个 epoch，并使用 10000 张生成图计算得到 `FID = 23.1820`。StyleGAN3 使用官方 FFHQ 预训练权重生成 1024 张高质量人像，并生成潜空间插值视频。由于 StyleGAN3 的训练域为 FFHQ，而本实验中用于 FID 对比的真实集为 CelebA_64，二者在人脸裁剪方式、分辨率和数据分布上均不一致，因此 StyleGAN3 的跨域 `FID = 216.2482` 只作为参考，不作为公平模型排序依据。实验结果表明：手写 DCGAN 能够学习 CelebA 的基本人脸分布，但在清晰度、稳定性和细节方面仍有限；StyleGAN3 在真实感、多样性和潜空间连续性方面显著优于基础 DCGAN；AnimeGANv2 和 CycleGAN 则提供了面向个人照片展示的应用扩展。

关键词：GAN；DCGAN；StyleGAN3；CelebA；FFHQ；AnimeGANv2；CycleGAN；FID

## 1. 研究背景与课程关联

生成式模型的目标是学习数据分布，并从该分布中采样生成新的样本。GAN 由生成器和判别器组成：生成器从随机噪声中合成样本，判别器判断输入是真实样本还是生成样本，二者通过对抗训练共同优化。Goodfellow 等人提出的 GAN 框架为图像生成任务提供了基础范式 [1]。在课程学习中，GAN 的关键难点包括训练不稳定、模式崩塌、评价指标不完全可靠，以及如何将对抗学习扩展到条件生成或域转换任务。

本项目的设计没有直接跳到复杂模型，而是围绕课程学习路径逐步展开：

1. **从基础 GAN 到可训练 baseline**：通过手写 DCGAN，把理论中的生成器、判别器、对抗损失和交替优化落到代码。
2. **从 baseline 到高质量 SOTA 复现**：通过 StyleGAN3 官方预训练权重，观察现代 GAN 相对 DCGAN 的结构升级和质量提升。
3. **从图像生成到风格迁移扩展**：通过 AnimeGANv2 与 CycleGAN，把 GAN 思想扩展到动漫化和非成对图像翻译。
4. **从定性展示到定量分析**：使用 FID 评价 DCGAN 在 CelebA_64 同域设置下的生成分布，同时解释跨域 FID 的局限。

因此，本项目的重点不是简单堆叠多个开源模型，而是构建一条逻辑自洽的学习路线：先能手写和训练，再能复现成熟模型，最后能解释结果和局限。

## 2. 数据来源、使用边界与伦理说明

### 2.1 CelebA：baseline 训练数据

CelebA 是本项目 DCGAN baseline 的主训练数据。实验使用官方 Align&Cropped 版本，共 `202599` 张对齐人脸图像。原始数据放置在 AutoDL 的 `data/raw/celeba`，随后通过 `scripts/prepare_celeba.py` 统一处理为 64x64 分辨率，保存到 `data/processed/celeba_64`。

使用策略如下：

| 项目 | 设置 |
|---|---|
| 数据集 | CelebA Align&Cropped |
| 图像数量 | 202599 |
| 用途 | DCGAN 从零训练 |
| 处理方式 | resize 到 64x64，归一化到 `[-1, 1]` |
| 数据来源 | CelebA 官方页面 [8] |

选择 CelebA 的原因是它标准、公开、易获取，并且包含大量对齐人脸，适合课程作业中的基础人像生成任务。

### 2.2 FFHQ 与 StyleGAN3 官方预训练权重

StyleGAN3 部分使用 NVIDIA 官方 FFHQ 预训练权重，而不是从零训练。完整训练 StyleGAN3 对时间、数据、显存和工程调参要求都更高，不适合作为两天作业中的主训练任务。使用官方权重可以把重点放在模型复现、结果分析和与 DCGAN 的代际对比上。

使用策略如下：

| 项目 | 设置 |
|---|---|
| 模型 | StyleGAN3 |
| 数据域 | FFHQ |
| 权重来源 | NVIDIA StyleGAN3 官方仓库 [4] |
| 用途 | 高质量真实人像生成、潜空间插值视频 |
| 训练方式 | 不从零训练，复现官方预训练权重 |

需要注意的是，FFHQ 和 CelebA 并不是完全相同的数据域。FFHQ 图像质量、分辨率和裁剪方式与 CelebA_64 不同，因此 StyleGAN3 与 CelebA_64 的 FID 只能作为跨域参考。

### 2.3 个人照片与隐私边界

个人照片只用于推理展示，不参与 DCGAN 或 StyleGAN3 训练。这样做有两个原因：一是个人照片数量少，不适合作为训练集；二是可以避免将隐私数据混入公开训练流程。报告中展示个人照片风格化结果时，应说明其用途仅为课程演示，不用于身份识别、认证或误导性传播。

### 2.4 增强模块输入

AnimeGANv2 和 CycleGAN 的输入由两部分组成：

1. 个人照片，用于展示真实照片的动漫化和风格迁移效果。
2. StyleGAN3 生成头像，用于补充样例数量，使报告展示更充分。

这种设计保证增强模块不依赖大量私有数据，同时能展示不同输入下的稳定性。

## 3. 技术路线总览

本项目的 pipeline 如下：

```text
CelebA Align&Cropped
        |
        v
CelebA_64 预处理
        |
        v
手写 DCGAN baseline
        |
        +--> 固定噪声训练过程图
        +--> DCGAN 生成样例
        +--> 同域 FID 定量评价

StyleGAN3 FFHQ 官方预训练权重
        |
        +--> 高质量真实人像生成
        +--> 潜空间插值视频
        +--> 作为 AnimeGANv2/CycleGAN 输入补充

个人照片 / StyleGAN3 生成头像
        |
        +--> AnimeGANv2 动漫化
        +--> CycleGAN Monet / Van Gogh / Ukiyoe 风格迁移
```

该路线对应三个层级：

| 层级 | 模块 | 核心作用 | 报告定位 |
|---|---|---|---|
| 基础层 | DCGAN | 自己实现并训练 GAN | 课程学习过程 |
| 复现层 | StyleGAN3 | 官方预训练高质量人像生成 | SOTA 复现与对比 |
| 扩展层 | AnimeGANv2 / CycleGAN | 动漫化与风格迁移 | 应用展示与课程扩展 |

## 4. 阶段一：手写 DCGAN baseline

### 4.1 设计动机

DCGAN 是基础 GAN 向卷积图像生成任务发展的经典结构。相较最原始的全连接 GAN，DCGAN 使用卷积和转置卷积，更适合图像数据。Radford 等人提出的 DCGAN 证明了卷积结构在无监督图像表征和生成任务中的有效性 [2]。本项目使用 DCGAN 作为 baseline，主要目的是体现课程学习过程，而不是追求最终最高画质。

### 4.2 模型结构

生成器接收 `100` 维随机噪声，通过多层转置卷积逐步上采样，输出 `3 x 64 x 64` 的 RGB 人脸图像：

```text
z: 100 x 1 x 1
ConvTranspose2d -> BatchNorm -> ReLU
ConvTranspose2d -> BatchNorm -> ReLU
ConvTranspose2d -> BatchNorm -> ReLU
ConvTranspose2d -> BatchNorm -> ReLU
ConvTranspose2d -> Tanh
output: 3 x 64 x 64
```

判别器接收真实图像或生成图像，通过多层卷积下采样，最后输出真假判别 logit：

```text
input: 3 x 64 x 64
Conv2d -> LeakyReLU
Conv2d -> BatchNorm -> LeakyReLU
Conv2d -> BatchNorm -> LeakyReLU
Conv2d -> BatchNorm -> LeakyReLU
Conv2d
output: real/fake logit
```

训练中使用 `BCEWithLogitsLoss`。判别器学习将真实图像判断为 1、生成图像判断为 0；生成器学习让判别器把生成图像判断为 1。该过程对应 GAN 的 min-max 博弈目标。

### 4.3 训练参数与工程实现

| 参数 | 设置 |
|---|---:|
| 数据集 | CelebA_64 |
| 训练图像数量 | 202599 |
| 图像分辨率 | 64x64 |
| epoch | 60 |
| batch size | 256 |
| latent dim | 100 |
| learning rate | 2e-4 |
| beta1 | 0.5 |
| workers | 8 |
| GPU | AutoDL NVIDIA vGPU-32GB |

工程上，DCGAN 部分保留为手写实现：

- `src/dcgan/models.py`：定义生成器、判别器和权重初始化。
- `src/dcgan/dataset.py`：读取 CelebA_64 图像并做归一化。
- `src/dcgan/train.py`：完成交替训练、日志记录、样例保存和 checkpoint 保存。
- `scripts/prepare_celeba.py`：将 CelebA 原图整理为训练所需分辨率。

这种实现方式能在报告中清楚说明代码结构，也能体现“不是只调用现成库”的学习过程。

### 4.4 训练过程观察

训练过程中固定一组随机噪声，每隔一定 epoch 保存生成图。图 1 展示了从第 1、10、20、30、40、50 到第 60 轮的变化。

![图1 DCGAN 固定噪声训练过程，从左到右依次为 epoch 1、10、20、30、40、50、60。](GAN_results_images/final_images_package/report/selected_figures/fig01_dcgan_progress.png)

可以观察到：

- 早期图像只有大致色块和人脸轮廓，五官位置不稳定。
- 中期开始出现较清晰的头发、肤色和脸部结构。
- 后期样本明显更像真实人脸，但仍存在眼睛、嘴巴和背景的局部伪影。

这符合 DCGAN baseline 的预期：它能学习基本分布，但模型容量、分辨率和训练稳定性都限制了最终质量。

### 4.5 DCGAN 生成样例

图 2 展示了多组 DCGAN 生成样例。

![图2 DCGAN 生成样例网格。](GAN_results_images/final_images_package/report/selected_figures/fig02_dcgan_generated_grids.png)

从结果看，DCGAN 已经能够生成多样化的人脸，包括不同发型、肤色、姿态和背景。但与真实照片相比，部分样本仍有明显问题：局部五官变形、脸部边缘不自然、背景混杂和纹理粗糙。这些问题说明 DCGAN 适合作为 baseline，但不适合作为最终高质量真实人像生成方案。

## 5. 阶段二：StyleGAN3 高质量真实人像生成复现

### 5.1 从 DCGAN 到 StyleGAN 的必要性

DCGAN 的生成器主要依赖逐层上采样，控制能力有限。StyleGAN 系列则使用 style-based generator，将 latent code 映射为逐层风格控制信号，从而提升图像质量、可控性和潜空间表达能力。StyleGAN3 进一步关注 alias-free 生成，使图像在连续变化时具有更稳定的几何一致性 [4]。

在课程作业中引入 StyleGAN3 的意义不是“替代手写 DCGAN”，而是形成代际对比：

- DCGAN 展示基础 GAN 如何从噪声学习人脸分布。
- StyleGAN3 展示现代 GAN 如何通过结构设计和大规模训练得到高质量人像。
- 两者结合能体现从课程理论到前沿复现的完整学习链路。

### 5.2 复现方式

本项目使用 NVIDIA StyleGAN3 官方仓库和官方 FFHQ 预训练权重进行生成。运行脚本为：

```text
scripts/run_stylegan3_generate.sh
scripts/run_stylegan3_video.sh
```

实验输出：

| 项目 | 结果 |
|---|---:|
| 生成图数量 | 1024 |
| 图像分辨率 | 1024x1024 |
| 输出目录 | `outputs/stylegan3/images` |
| 插值视频 | `outputs/stylegan3/latent_interpolation.mp4` |

### 5.3 StyleGAN3 生成结果

图 3 展示了 StyleGAN3 生成的多张真实人像样例。

![图3 StyleGAN3 官方 FFHQ 预训练权重生成的人像样例。](GAN_results_images/final_images_package/report/selected_figures/fig03_stylegan3_samples.png)

与 DCGAN 相比，StyleGAN3 的提升非常明显：

- 皮肤、头发、眼睛、牙齿和服饰细节更真实。
- 光照、背景、年龄、性别和姿态变化更自然。
- 图像整体接近真实摄影质量，而不是低分辨率纹理拼接。
- 样本之间多样性更强，更适合做报告展示和后续风格化输入。

### 5.4 潜空间插值结果

StyleGAN3 还生成了潜空间插值视频：

```text
GAN_results_images/final_images_package/outputs/stylegan3/latent_interpolation.mp4
```

该视频展示了不同 latent code 之间的连续变化。人脸身份、表情、发型和光照能够平滑过渡，说明 StyleGAN3 的 latent space 学到的是连续的人脸分布，而不是简单记忆训练样本。这一点是基础 DCGAN 难以直观展示的。

## 6. 阶段三：AnimeGANv2 动漫化扩展

### 6.1 扩展动机

真实人像生成是主任务，但课程作业展示通常需要更直观的应用效果。AnimeGANv2 可以将真实或生成的人像转换为动漫风格，适合展示个人照片或 StyleGAN3 生成头像的风格化效果。该模块不参与主线 FID 评价，只作为应用扩展。

### 6.2 输入与风格设置

由于个人照片数量有限，实验将个人照片与 StyleGAN3 生成头像共同作为输入，整理出 36 张增强输入。测试的风格包括：

- `face_paint_v2`
- `face_paint_v1`
- `paprika`

图 4 展示了多组输入与不同 AnimeGANv2 风格的对比。每一组从左到右依次为原图、`face_paint_v2`、`face_paint_v1`、`paprika`。

![图4 AnimeGANv2 多风格动漫化对比。](GAN_results_images/final_images_package/report/selected_figures/fig04_animegan2_comparison.png)

### 6.3 结果分析

AnimeGANv2 的效果集中体现在以下方面：

- 眼睛和面部轮廓更接近动漫头像。
- 皮肤纹理被平滑化，整体更干净。
- 发丝和阴影被重新绘制，风格感更强。

局限也比较明显：

- 身份特征会被弱化。
- 真实皮肤纹理和年龄特征会被过度平滑。
- 对复杂背景、遮挡和侧脸输入更敏感。

因此，AnimeGANv2 适合写作“增强展示”，不应被描述为真实人像生成主模型。

## 7. 阶段四：CycleGAN 风格迁移扩展

### 7.1 课程关联

CycleGAN 是非成对图像翻译的经典模型。它通过两个方向的生成器和判别器学习域间映射，并使用 cycle consistency loss 约束图像在 A→B→A 后能够回到原域 [5]。这使模型不需要严格成对的训练样本，适合风格迁移、季节转换、照片与绘画转换等任务。

本项目使用官方 CycleGAN/pix2pix 仓库中的预训练风格模型进行推理，包括：

- `style_vangogh`
- `style_monet`
- `style_ukiyoe`

### 7.2 风格迁移结果

图 5 展示了原图与 Ukiyoe 风格迁移结果的对比。

![图5 CycleGAN Ukiyoe 风格迁移对比。](GAN_results_images/final_images_package/report/selected_figures/fig05_cyclegan_ukiyoe_comparison.png)

结果显示，CycleGAN 能够显著改变图像的色彩、线条和整体纹理，使输入头像具有明显艺术风格。与 AnimeGANv2 相比，CycleGAN 的风格迁移更偏向整体色彩和纹理域转换，而不是专门优化二次元人脸特征。

### 7.3 局限

CycleGAN 的主要局限是身份保持能力有限。风格越强，原始人脸的细节越容易被改变，例如肤色、嘴唇、鼻梁和头发边缘可能发生偏移。因此报告中将 CycleGAN 定位为“呼应课程重点的增强模块”，而不是“真实人像生成主模型”。

## 8. 定量评价：FID 与结果解释

### 8.1 FID 指标含义

FID（Fréchet Inception Distance）通过比较真实图像和生成图像在 Inception 特征空间中的均值和协方差来衡量分布距离。Heusel 等人提出的 FID 已成为 GAN 评价中常用指标之一 [6]。一般来说，FID 越低，表示生成分布越接近真实分布。

但 FID 有重要前提：比较双方应来自相同或高度一致的数据域，并使用一致的分辨率和预处理方式。跨数据集、跨分辨率或跨裁剪策略的 FID 容易产生误导。

### 8.2 本项目 FID 结果

| 模型 | 真实集 | 生成集 | 样本数 | FID | 用途 |
|---|---|---|---:|---:|---|
| DCGAN | CelebA_64 | DCGAN 生成图 | 10000 | 23.1820 | 主定量指标 |
| DCGAN 初次评估 | CelebA_64 | DCGAN 生成图 | 4096 | 25.4784 | 辅助记录 |
| StyleGAN3 | CelebA_64 | StyleGAN3 FFHQ 生成图 | 1024 | 216.2482 | 跨域参考 |

DCGAN 的 `FID = 23.1820` 是本项目最重要的定量结果，因为真实集和生成集都处于 CelebA_64 域，评价设置相对一致。StyleGAN3 的 `FID = 216.2482` 不能用于说明 StyleGAN3 比 DCGAN 差，因为 StyleGAN3 生成图来自 FFHQ 域，真实集却是 CelebA_64，二者并不满足公平比较条件。

正确结论应写为：

```text
DCGAN 在同域 CelebA_64 设置下取得 FID = 23.1820，说明 baseline 已经学习到基本人脸分布。StyleGAN3 使用 FFHQ 官方预训练权重，视觉质量显著优于 DCGAN；其与 CelebA_64 的 FID 属于跨域参考，不作为公平排序依据。
```

## 9. 结果素材统计与使用方式

本地解压后的结果包位于：

```text
C:\GAN\GAN_results_images\final_images_package
```

检查结果如下：

| 类别 | 数量或状态 |
|---|---:|
| PNG 图片 | 7634 |
| MP4 视频 | 2 |
| FID/说明文本 | 7 |
| Markdown 文档 | 7 |
| 模型权重 `.pth/.pt/.pkl/.ckpt` | 0 |
| StyleGAN3 生成图 | 1024 |
| DCGAN 报告图与样例 | 248 |
| AnimeGANv2 对比素材 | 108 |
| Anime 报告精选素材 | 180 |
| CycleGAN Ukiyoe fake 图 | 36 |

报告中建议使用如下素材：

| 图编号 | 文件 | 用途 |
|---|---|---|
| 图1 | `report/selected_figures/fig01_dcgan_progress.png` | 展示 DCGAN 训练递进过程 |
| 图2 | `report/selected_figures/fig02_dcgan_generated_grids.png` | 展示 DCGAN baseline 生成能力 |
| 图3 | `report/selected_figures/fig03_stylegan3_samples.png` | 展示 StyleGAN3 高质量人像 |
| 图4 | `report/selected_figures/fig04_animegan2_comparison.png` | 展示 AnimeGANv2 多风格动漫化 |
| 图5 | `report/selected_figures/fig05_cyclegan_ukiyoe_comparison.png` | 展示 CycleGAN 风格迁移 |

## 10. 失败现象与反思

### 10.1 DCGAN 的失败现象

DCGAN 后期仍存在局部伪影，主要包括：

- 眼睛和嘴巴位置可能不对称。
- 发际线和背景边界有时混在一起。
- 侧脸、遮挡和复杂发型生成不稳定。
- 继续训练不一定稳定降低 FID，可能出现震荡。

这些问题不是实验失败，而是基础 GAN 的合理局限。它们说明生成模型质量不仅依赖训练轮数，还依赖网络结构、损失设计、归一化方式、数据分辨率和训练稳定性。

### 10.2 StyleGAN3 指标局限

StyleGAN3 的视觉质量明显更高，但本项目没有下载完整 FFHQ 真实集，也没有使用 StyleGAN3 官方 metric 工具计算 FFHQ 同域 FID。因此 StyleGAN3 的定量评价部分存在局限。更严格的后续实验应使用 FFHQ 真实集或官方 metric 工具进行同域评价。

### 10.3 动漫化与风格迁移局限

AnimeGANv2 和 CycleGAN 的输出适合展示，但不适合做人脸身份保持评价。强风格化会改变脸型、肤色、五官和纹理，因此这两个模块应定位为应用扩展，而不是主线生成模型。

## 11. 工程复现总结

本项目在 AutoDL 上完成了从数据准备到结果打包的完整流程：

1. 上传项目工程、CelebA 数据集压缩包和个人照片压缩包。
2. 解压项目并检查 PyTorch、CUDA、StyleGAN3、AnimeGANv2、CycleGAN 环境。
3. 将 CelebA 处理为 `CelebA_64`。
4. 训练 DCGAN baseline，并保存样例、日志和 checkpoint。
5. 使用 StyleGAN3 官方权重生成 1024 张高质量人像和插值视频。
6. 使用 AnimeGANv2 运行多种动漫化风格。
7. 使用 CycleGAN 运行 Van Gogh、Monet 和 Ukiyoe 风格迁移。
8. 计算 DCGAN FID，并解释 StyleGAN3 跨域 FID。
9. 收集报告素材，最终打包为不含权重的轻量结果包。

该流程体现了课程作业所需要的完整性：有模型设计、有代码实现、有公开数据、有复现结果、有指标分析、有个人照片展示，也有对失败和局限的反思。

## 12. 伦理与合规说明

1. CelebA、FFHQ、StyleGAN3、CycleGAN 和 AnimeGANv2 均仅用于课程学习与非商业实验展示。
2. 个人照片不参与模型训练，只用于本地推理展示。
3. 本项目不用于人脸识别、身份认证、伪造真实身份或误导性传播。
4. 报告中展示生成图时，应说明图片由 GAN 生成或风格化得到，避免被误解为真实人物照片。

## 13. 结论

本项目完成了一条从基础 GAN 到高质量 GAN，再到风格迁移扩展的完整技术路线。DCGAN 部分体现了课程中 GAN 对抗训练的核心机制，在 CelebA_64 上训练后得到 `FID = 23.1820`，能够生成基本真实的人脸图像；StyleGAN3 部分复现了官方 FFHQ 预训练模型，生成结果在分辨率、真实感和细节表达上显著优于 DCGAN；AnimeGANv2 与 CycleGAN 则将人像生成结果扩展到动漫化和艺术风格迁移场景。

从学习角度看，本项目的价值不只在于得到若干生成图，而在于形成了清晰的递进关系：基础模型用于理解 GAN 原理，SOTA 复现用于观察现代 GAN 能力，增强模块用于连接课程中的 CycleGAN 知识点，FID 和失败样例用于支撑客观分析。后续若继续改进，可以尝试更稳定的 GAN 训练策略、StyleGAN 官方同域指标计算、StyleGAN projector 个人照片投影，以及更强的人脸身份保持风格迁移模型。

## 参考文献与开源来源

[1] I. Goodfellow et al., "Generative adversarial nets," in *Advances in Neural Information Processing Systems*, 2014.

[2] A. Radford, L. Metz, and S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," arXiv:1511.06434, 2015.

[3] T. Karras et al., "Analyzing and improving the image quality of StyleGAN," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition*, 2020.

[4] T. Karras et al., "Alias-free generative adversarial networks," in *Advances in Neural Information Processing Systems*, 2021. [Online]. Available: https://github.com/NVlabs/stylegan3

[5] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in *Proc. IEEE Int. Conf. Computer Vision*, 2017. [Online]. Available: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

[6] M. Heusel et al., "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in *Advances in Neural Information Processing Systems*, 2017.

[7] PyTorch, "DCGAN Tutorial." [Online]. Available: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

[8] The Chinese University of Hong Kong Multimedia Laboratory, "Large-scale CelebFaces Attributes (CelebA) Dataset." [Online]. Available: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

[9] NVIDIA, "Flickr-Faces-HQ Dataset." [Online]. Available: https://github.com/NVlabs/ffhq-dataset

[10] B. Lee, "animegan2-pytorch." [Online]. Available: https://github.com/bryandlee/animegan2-pytorch

[11] M. Seitzer, "pytorch-fid." [Online]. Available: https://github.com/mseitzer/pytorch-fid

