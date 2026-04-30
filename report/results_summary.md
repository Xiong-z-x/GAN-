# 实验结果汇总

更新时间：2026-04-30

## 1. DCGAN baseline

| 项目 | 结果 |
|---|---|
| 数据集 | 未运行 |
| 分辨率 | 64x64 |
| epoch | 未运行 |
| batch size | 未运行 |
| 生成样例 | 未生成 |
| loss 日志 | 未生成 |

## 2. StyleGAN 官方预训练生成

| 项目 | 结果 |
|---|---|
| 模型 | StyleGAN3 FFHQ 预训练 |
| 权重 | 未运行 |
| 随机种子 | 未运行 |
| 生成样例 | 未生成 |
| 插值视频 | 未生成 |

默认脚本：

```bash
bash scripts/run_stylegan3_generate.sh
bash scripts/run_stylegan3_video.sh
```

## 3. AnimeGANv2 动漫化

| 项目 | 结果 |
|---|---|
| 输入 | 未运行 |
| 权重 | 未运行 |
| 输出 | 未生成 |

默认脚本：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
bash scripts/run_animegan2_infer.sh
```

## 4. CycleGAN 风格迁移

| 项目 | 结果 |
|---|---|
| 预训练模型 | 未运行 |
| 输入 | 未运行 |
| 输出 | 未生成 |

默认脚本：

```bash
python scripts/prepare_style_transfer_inputs.py --clear
bash scripts/run_cyclegan_pretrained_style.sh
```

## 5. 指标

| 指标 | DCGAN | StyleGAN |
|---|---:|---:|
| FID | 未计算 | 未计算 |
| KID | 未计算 | 未计算 |
| Precision | 未计算 | 未计算 |
| Recall | 未计算 | 未计算 |

## 6. 待补充结论

- 待填入 DCGAN 与 StyleGAN 的视觉差异。
- 待填入动漫化与风格迁移的展示结论。
- 待填入失败案例。
