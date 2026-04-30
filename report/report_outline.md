# 实验报告骨架

## 1. 研究背景与课程关联

- 说明本项目围绕 GAN 真实人像生成展开。
- 说明课程已学习 GAN basic、WGAN、GAN evaluation、CycleGAN。
- 说明项目从手写 DCGAN 过渡到 StyleGAN 官方预训练复现，再扩展到动漫化和风格迁移。

## 2. 数据集与预处理

- CelebA：用于 DCGAN baseline。
- FFHQ：作为 StyleGAN 官方预训练模型的人脸域。
- 个人照片：只用于 demo，不作为训练集。
- 可选 MetFaces：用于艺术风格迁移附录。

## 3. Baseline：手写 DCGAN

- 生成器结构。
- 判别器结构。
- 对抗损失与交替优化。
- 训练参数。
- 生成结果和失败现象。

## 4. 高质量真实人像生成：StyleGAN 官方预训练复现

- 说明 StyleGAN3/StyleGAN2 的角色。
- 展示随机生成人像。
- 展示潜空间插值视频。
- 可选展示 StyleGAN2-ADA projector。

## 5. 增强模块：动漫化与风格迁移

- AnimeGANv2 预训练推理。
- CycleGAN 官方预训练风格迁移。
- 说明这些模块是应用扩展，不替代真实人像生成主线。

## 6. 实验结果与指标

- DCGAN loss 曲线。
- DCGAN 与 StyleGAN 生成图对比。
- FID/KID/PR，如时间允许。
- 动漫化和风格迁移主观质量分析。

## 7. 个人照片示例

- 原始照片。
- 动漫化结果。
- 风格迁移结果。
- 可选 projector 结果。

## 8. 失败案例分析

- DCGAN 模糊或模式崩塌。
- 动漫化对遮挡、侧脸、复杂背景的失败。
- 风格迁移导致颜色漂移或身份弱化。

## 9. 伦理、许可与局限

- CelebA、FFHQ、MetFaces 的非商业研究边界。
- 个人照片只用于课程展示。
- 不用于人脸识别技术开发。

## 10. 结论

- 总结从手写 GAN 到官方预训练高质量生成的提升。
- 总结 CycleGAN/AnimeGANv2 增强模块的价值。
- 给出后续可改进方向。

