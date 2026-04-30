# 任务计划

项目：GAN 真实人像生成大作业  
当前阶段：技术路线已按新约束更新，准备进入代码实现前的目录与环境搭建。

## 目标

在两天内完成一个可提交的大作业项目：包含设计文档、实验结果、实验报告、代码说明；核心围绕 GAN，主任务是真实人像生成，动漫化/风格迁移作为增强模块，并显式结合 CycleGAN 课程重点。

## 阶段状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| 1. 参考文件摄取 | complete | 已读取 Gemini PDF 与 GPT Markdown |
| 2. 联网真实性核验 | complete | 已核验课程页、模型仓库、数据集与权重 |
| 3. 方案综合 | complete | 最终路线：DCGAN + StyleGAN 官方预训练生成 + AnimeGANv2/CycleGAN 增强 |
| 4. 文档固化 | complete | `技术路线_最终方案.md`、`项目完成进度.md` |
| 5. AutoDL 配置纳入 | complete | 已按 RTX 4090D 24GB、CUDA 12.8、PyTorch 2.8、50GB 数据盘规划 |
| 6. 代码实现 | complete | 已创建 DCGAN 源码、数据脚本、外部模型封装脚本 |
| 7. SOTA/增强模块工程化 | complete | 已创建 AutoDL 一键流水线、输入准备、条件检查和输出检查脚本 |
| 8. 实验运行 | pending | 训练、采样、投影、CycleGAN 结果 |
| 9. 报告交付 | pending | 已创建报告骨架和结果汇总模板，仍需真实实验结果 |

## 关键决策

- 主线是真实人像生成；动漫化/风格迁移作为增强模块。
- Baseline 采用手写 DCGAN。
- SOTA 主线优先采用 StyleGAN3 官方 FFHQ 预训练；StyleGAN2-ADA projector 可选。
- AnimeGANv2 用于个人照片动漫化，优先预训练推理。
- CycleGAN 作为课程重点模块，优先使用官方风格迁移预训练模型；自训只做加分项。
- 自拍只用于 demo、动漫化、风格迁移和可选 projector，不用于主训练。

## 下一步

1. 推荐直接运行 `scripts/run_sota_enhanced_pipeline.sh` 复现 StyleGAN3、AnimeGANv2 和 CycleGAN。
2. 准备 CelebA 数据和个人照片。
3. 运行 `src.dcgan.train` 训练 baseline。
4. 汇总报告素材并补充实验结论。
