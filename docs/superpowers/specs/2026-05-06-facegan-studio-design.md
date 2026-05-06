# FaceGAN Studio 程序设计方案

日期：2026-05-06

## 1. 设计目标

本程序作为当前 GAN 课程项目的最终应用封装层，目标是在保留现有全部实验成果的基础上，提供一个可交互的人脸生成与风格化系统。用户上传一张人脸后，可以选择不同任务模式，生成动漫风、证件照、戴眼镜造型、轻度美化和不同姿态结果。

本程序不把所有功能都强行归为 DCGAN。技术边界如下：

- DCGAN / DCGAN++：作为课程基础 GAN 训练与能力边界展示。
- StyleGAN3：作为成熟 GAN 的高质量人像生成上限展示。
- AnimeGANv2 / CycleGAN：作为 GAN 风格迁移模块，用于动漫风和艺术风格转换。
- InstantID：作为身份保持生成模块，用于输入人脸的姿态、造型和场景变化。
- 证件照模块：以人脸检测、对齐、裁剪、背景替换和轻度增强为主，必要时再接身份保持生成。

核心原则：输入人脸编辑任务必须优先保持身份一致，不能为了生成效果牺牲原始长相。

## 2. 当前已有成果边界

本程序会复用当前项目已有成果，但不覆盖原始目录。

已确认的新成果目录：

```text
GAN_new_showcase_results/
  outputs/gan_showcase/
  outputs/instantid_myface_pose/
  report/report_assets/gan_showcase/
  report/report_assets/instantid_myface_pose/
```

已确认素材：

| 模块 | 素材 | 作用 |
|---|---|---|
| DCGAN 演化 | `dcgan_evolution_grid.png` | 展示 E0/E1/E2/续训/早期 DCGAN 的对比 |
| StyleGAN3 上限 | `stylegan3_top16_grid.png`, `stylegan3_top32_grid.png` | 展示成熟 GAN 的高质量人像生成能力 |
| InstantID 姿态参考 | `pose_reference_grid_4x4.png` | 展示姿态来源 |
| InstantID 身份生成 | `my_face_pose_grid_4x4.png` | 展示“我的脸 + 不同姿态/场景”的生成结果 |
| 说明文件 | `gan_showcase_summary.txt`, `instantid_showcase_summary.txt` | 写入报告的数据来源说明 |

已有报告结构保留，但需要新增“FaceGAN Studio 应用封装”章节，把交互式程序作为最终综合成果。

## 3. 用户功能

### 3.1 模式一：动漫风格化

输入：一张人脸图片。

输出：

- Face Paint v2 动漫风
- Face Paint v1 动漫风
- Paprika 动漫风
- 可选艺术风格：Van Gogh / Monet / Ukiyoe

处理流程：

1. 上传图片。
2. 自动检测人脸，选择最大可见人脸。
3. 对输入图做 RGB 统一、尺寸标准化和必要裁剪。
4. 调用 AnimeGANv2 三个权重生成三种动漫风。
5. 可选调用 CycleGAN 生成艺术风格图。
6. 生成单图和对比拼图。
7. 保存到程序输出目录和报告素材目录。

身份保持策略：

- 动漫化本身会改变面部线条，因此只承诺“保留大致脸型和五官布局”，不承诺像 InstantID 一样稳定保留身份。
- 报告中应明确这是风格迁移实验，不是身份保持实验。

### 3.2 模式二：证件照生成

输入：一张人脸图片。

输出：

- 白底证件照
- 蓝底证件照
- 红底证件照
- 可选职业正装版

处理流程：

1. 检测人脸和关键点。
2. 进行人脸对齐。
3. 按证件照比例裁剪成头肩构图。
4. 分割或近似替换背景。
5. 输出白、蓝、红三种纯色背景。
6. 进行轻度亮度修正、肤色平衡、锐化和压缩尺寸导出。
7. 生成证件照对比拼图。

身份保持策略：

- 默认不大幅修改五官，不做夸张磨皮。
- 如果做正装版，应标注为“生成式扩展版本”，不作为正式证件照审核用途。

### 3.3 模式三：造型与姿态生成

输入：一张人脸图片。

输出：

- 戴黑框眼镜
- 戴金属框眼镜
- 职业照
- 休闲照
- 侧脸姿态
- 微笑姿态
- 半身照
- 电影感肖像

处理流程：

1. 使用 InsightFace 检测输入人脸。
2. 提取输入人脸身份向量。
3. 从姿态参考库中选择关键点参考。
4. 使用 InstantID 固定身份向量。
5. 使用不同 prompt 控制造型、眼镜、服装、姿态和场景。
6. 生成 8 到 16 张结果。
7. 生成 4x4 拼图和单图。
8. 保存输出和报告素材。

身份保持策略：

- 固定输入人脸 embedding。
- 默认 `ip_adapter_scale` 设置偏高，优先身份一致。
- prompt 避免“完全改变发型、年龄、人种”等强身份扰动描述。
- 对每张结果保留输入图和姿态参考图，便于报告解释生成来源。

## 4. 页面设计

程序推荐使用 Gradio 本地 Web 应用。

### 4.1 页面一：输入区

组件：

- 图片上传框
- 原图预览
- 人脸检测预览
- 当前检测状态

校验：

- 如果没有检测到人脸，拒绝生成并提示重新上传。
- 如果检测到多张人脸，默认选择面积最大的人脸，并提示用户。

### 4.2 页面二：功能选择区

功能按钮：

- 动漫风格化
- 证件照生成
- 造型与姿态生成
- 项目成果展示

### 4.3 页面三：参数区

动漫风格化参数：

- 风格选择：Face Paint v2 / Face Paint v1 / Paprika / 全部
- 是否追加 CycleGAN 艺术风格

证件照参数：

- 背景颜色：白 / 蓝 / 红 / 全部
- 输出尺寸：一寸 / 二寸 / 方形头像
- 是否轻度美化

造型与姿态参数：

- 生成数量：4 / 8 / 16
- 造型模板：眼镜 / 职业 / 休闲 / 电影感 / 全部
- 身份保持强度：标准 / 更像本人
- 姿态来源：内置参考 / 上传参考 / 混合参考

### 4.4 页面四：输出区

组件：

- 结果图库
- 对比拼图
- 输出目录显示
- 复制到报告素材目录按钮

### 4.5 页面五：项目成果展示

展示现有成果：

- DCGAN 演化图
- StyleGAN3 高质量生成图
- AnimeGANv2 三风格图
- CycleGAN 艺术风格图
- InstantID 个人身份保持生成图

这一页用于课程展示，不参与新图生成。

## 5. 工程结构

建议新增目录：

```text
facegan_studio/
  app.py
  config.py
  modules/
    face_detector.py
    anime_style.py
    id_photo.py
    pose_styler.py
    gallery.py
    image_utils.py
    paths.py
  presets/
    anime_styles.json
    id_photo_templates.json
    pose_prompts.json
  README.md
```

输出目录：

```text
outputs/facegan_studio/
  anime/
  id_photo/
  pose_style/
  showcase/
  grids/
```

报告素材同步目录：

```text
report/report_assets/facegan_studio/
  anime/
  id_photo/
  pose_style/
  showcase/
  grids/
```

## 6. 模块职责

### 6.1 `paths.py`

职责：

- 管理项目根目录。
- 管理 AutoDL 与本地路径差异。
- 避免脚本里硬编码过多路径。

关键路径：

- `external/animegan2-pytorch`
- `external/pytorch-CycleGAN-and-pix2pix`
- `external/InstantID`
- `outputs/facegan_studio`
- `report/report_assets/facegan_studio`

### 6.2 `face_detector.py`

职责：

- 使用 InsightFace 检测人脸。
- 选择最大人脸。
- 提取 bounding box、关键点和身份 embedding。
- 保存检测预览图。

失败处理：

- 没有人脸：终止当前任务。
- 多张人脸：选择最大脸，并在界面提示。
- 人脸太小：提示用户换清晰照片。

### 6.3 `anime_style.py`

职责：

- 准备 AnimeGANv2 输入目录。
- 调用 AnimeGANv2 推理脚本或直接调用模型。
- 汇总三种权重输出。
- 可选调用 CycleGAN 风格迁移。

输出：

- 单风格图片
- 多风格拼图

### 6.4 `id_photo.py`

职责：

- 人脸对齐。
- 证件照裁剪。
- 背景替换。
- 亮度和肤色轻度修正。
- 输出常用证件照尺寸。

默认策略：

- 证件照模块尽量少生成式修改，以保持原貌。
- 正装版本作为可选扩展。

### 6.5 `pose_styler.py`

职责：

- 加载 InstantID。
- 加载本地 SDXL 基础模型。
- 加载 ControlNetModel 和 ip-adapter。
- 使用输入人脸身份 embedding。
- 按 preset prompt 批量生成不同造型。

关键要求：

- 默认 `local_files_only=True`，避免运行时联网下载。
- 输出目录每次按时间戳分开，避免覆盖前次结果。
- 每张图记录 prompt、seed、输入图和参考姿态来源。

### 6.6 `gallery.py`

职责：

- 读取已有成果目录。
- 生成项目展示页。
- 汇总 DCGAN、StyleGAN3、AnimeGANv2、CycleGAN、InstantID 成果。

### 6.7 `image_utils.py`

职责：

- 图像读取、RGB 转换、缩放、拼图。
- 保存图片和说明 JSON。
- 生成报告用网格图。

## 7. 模型与依赖边界

### 7.1 必需模型

| 功能 | 模型或仓库 | 状态 |
|---|---|---|
| 动漫风格化 | AnimeGANv2 | 已在项目外部仓库使用过 |
| 艺术风格 | CycleGAN | 已在项目外部仓库使用过 |
| 身份保持 | InstantID | 已跑通过同脸不同姿态成果 |
| 基础生成 | YamerMIX_v8 / SDXL | 已下载到本地缓存 |
| 人脸检测 | InsightFace antelopev2 | 已验证可用 |

### 7.2 不建议新引入的内容

第一版不引入新大模型，不再大量下载 Hugging Face 权重。原因：

- 当前成果已经足够支撑课程展示。
- AutoDL 环境已经经历过依赖冲突，继续加模型风险高。
- 新增功能的核心价值在封装与流程，而不是再次堆模型。

### 7.3 可选增强

如果后续确实需要更强人脸修复，可以再接：

- GFPGAN
- CodeFormer

但第一版设计中只作为可选项，不作为主路径。

## 8. 数据与伦理边界

程序必须在界面和报告中保留以下说明：

1. 个人照片只用于本人实验展示。
2. 不将个人照片并入公开训练集。
3. InstantID 生成图是合成图，不应用于身份认证、证件审核或误导性场景。
4. 证件照模块输出只作为课程技术演示，不保证符合官方证件照审核规范。
5. StyleGAN3、CelebA、FFHQ、AnimeGANv2、CycleGAN、InstantID 的来源需要在报告中列出。

## 9. 输出命名规范

每次运行使用时间戳：

```text
outputs/facegan_studio/<mode>/<YYYYMMDD_HHMMSS>/
```

示例：

```text
outputs/facegan_studio/anime/20260506_210000/
  input.png
  face_detected.png
  anime_face_paint_v2.png
  anime_face_paint_v1.png
  anime_paprika.png
  anime_grid.png
  metadata.json
```

姿态生成示例：

```text
outputs/facegan_studio/pose_style/20260506_211000/
  input.png
  identity_reference.png
  pose_reference_grid.png
  result_00_glasses_black.png
  result_01_glasses_metal.png
  result_02_business.png
  result_grid.png
  metadata.json
```

报告素材同步：

```text
report/report_assets/facegan_studio/<mode>/<YYYYMMDD_HHMMSS>/
```

## 10. 报告整合方案

最终报告新增一章：

```text
阶段六：FaceGAN Studio 人脸风格化应用封装
```

章节结构：

1. 应用设计动机
2. 为什么 DCGAN 不适合输入人脸编辑
3. 系统总体架构
4. 动漫风格化模块
5. 证件照生成模块
6. 造型与姿态生成模块
7. 输出结果展示
8. 局限性与伦理说明

新增图：

| 图 | 内容 |
|---|---|
| FaceGAN Studio 页面截图 | 展示程序界面 |
| 动漫风格化输出拼图 | 展示三种 AnimeGANv2 风格 |
| 证件照输出拼图 | 展示白底、蓝底、红底 |
| 造型与姿态输出拼图 | 展示眼镜、职业、休闲、侧脸等 |
| 项目成果展示页 | 汇总 DCGAN、StyleGAN3、InstantID |

## 11. 运行方式设计

AutoDL 运行：

```bash
cd /root/autodl-tmp/GAN
python -m facegan_studio.app --host 0.0.0.0 --port 7860
```

本地只做查看和报告整理，不建议在本地 CPU 上运行 InstantID。

## 12. 验收标准

第一版程序完成后必须满足：

1. 能启动 Gradio 页面。
2. 能上传一张人脸并检测出人脸。
3. 动漫风格化至少输出三张结果和一张拼图。
4. 证件照至少输出白底、蓝底、红底三张结果和一张拼图。
5. 造型与姿态生成至少输出 8 张结果和一张拼图。
6. 所有结果保存到 `outputs/facegan_studio/`。
7. 报告素材同步到 `report/report_assets/facegan_studio/`。
8. 项目成果展示页能读取当前已有成果。
9. 不覆盖 `GAN_new_showcase_results/` 和旧报告素材。
10. 失败时给出明确错误，不静默生成空文件。

## 13. 实施计划

### 13.1 第一阶段：工程骨架

- 新建 `facegan_studio/` 包。
- 新建路径管理和输出目录管理。
- 新建 Gradio 页面。
- 实现上传、预览、人脸检测。

### 13.2 第二阶段：项目成果展示页

- 读取 `GAN_new_showcase_results/report/report_assets/`。
- 展示 DCGAN 演化、StyleGAN3、InstantID 现有成果。
- 保证不修改原始素材。

### 13.3 第三阶段：动漫风格化模块

- 接入 AnimeGANv2 三权重。
- 生成三张动漫风结果。
- 生成对比拼图。
- 保存 metadata。

### 13.4 第四阶段：证件照模块

- 实现人脸裁剪和背景替换。
- 输出白、蓝、红底。
- 实现轻度亮度和锐化增强。

### 13.5 第五阶段：造型与姿态模块

- 接入 InstantID 本地模型路径。
- 使用输入人脸身份 embedding。
- 使用内置 prompt 生成不同造型。
- 输出结果拼图和 metadata。

### 13.6 第六阶段：报告整合

- 复制结果到报告素材目录。
- 更新报告图片索引。
- 新增 FaceGAN Studio 章节。
- 重新生成 Word / PDF。

## 14. 风险与应对

| 风险 | 应对 |
|---|---|
| InstantID 身份漂移 | 提高身份保持强度，减少强风格 prompt |
| 眼镜生成不稳定 | 生成多张候选，人工筛选最佳 |
| 证件照背景分割不准 | 第一版使用人脸对齐和保守裁剪，避免复杂背景替换失败 |
| AutoDL 依赖冲突 | 不重新安装 torch，不升级核心依赖 |
| 运行时联网失败 | 所有大模型默认本地路径加载 |
| 输出覆盖旧成果 | 使用时间戳目录，报告素材单独同步 |

## 15. 最终定位

FaceGAN Studio 不是把 DCGAN 包装成万能应用，而是把课程中的不同生成技术组织成一个统一的人脸生成系统：

- DCGAN 负责解释 GAN 基础训练过程。
- StyleGAN3 负责展示高质量 GAN 上限。
- AnimeGANv2 / CycleGAN 负责展示 GAN 风格迁移。
- InstantID 负责完成输入人脸的身份保持姿态与造型生成。

这样既保留课程技术主线，也能形成可展示、可交互、可写入报告的最终作品。
