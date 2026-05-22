# FaceGAN Studio 运行说明

## 1. 同步到 AutoDL

当前推荐通过 GitHub 同步源码。Remote-SSH 登录 AutoDL 后：

```bash
cd /root/autodl-tmp/GAN
git status --short
git pull --ff-only
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

如果 AutoDL 上还没有项目目录：

```bash
cd /root/autodl-tmp
git clone https://github.com/Xiong-z-x/GAN-.git GAN
cd /root/autodl-tmp/GAN
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

旧的压缩包传输方式只作为备用。如果 AutoDL 上已经有 `/root/autodl-tmp/GAN` 项目目录，只需要同步以下新增或修改文件：

```text
facegan_studio/
scripts/run_facegan_studio.sh
scripts/run_cyclegan_pretrained_style.sh
requirements_autodl.txt
README.md
代码附录.md
```

如果你要打一个只包含 FaceGAN Studio 的压缩包，在 Windows 项目根目录执行：

```powershell
Compress-Archive -Force `
  facegan_studio, `
  scripts/run_facegan_studio.sh, `
  scripts/run_cyclegan_pretrained_style.sh, `
  requirements_autodl.txt, `
  README.md, `
  代码附录.md `
  FaceGAN_Studio_code.zip
```

上传到 AutoDL 后解压：

```bash
cd /root/autodl-tmp/GAN
unzip -o /root/autodl-tmp/FaceGAN_Studio_code.zip -d /root/autodl-tmp/GAN
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

## 2. 安装依赖

不要重装 PyTorch。只补 Gradio 和项目依赖：

```bash
cd /root/autodl-tmp/GAN
pip install -r requirements_autodl.txt
```

如果你担心依赖影响已有环境，可以只安装 Gradio：

```bash
pip install gradio
```

## 3. 启动程序

```bash
cd /root/autodl-tmp/GAN
bash scripts/run_facegan_studio.sh
```

默认端口是 `7860`。如果需要换端口：

```bash
FACEGAN_PORT=7861 bash scripts/run_facegan_studio.sh
```

## 4. 输出目录

```text
outputs/facegan_studio/
report/report_assets/facegan_studio/
```

## 5. 功能说明

- 输入预览：检测上传图中的人脸。
- 动漫风格化：调用 AnimeGANv2 三种权重，可选 CycleGAN Van Gogh。
- 造型与姿态：调用 InstantID / InstantID img2img 做身份保持生成；该功能可能带来写真化或身份漂移，结果必须人工筛选。
- 轻造型保脸：读取 `data/raw/my_photos/` 中 4 张本人照片，按 InsightFace/InstantID 人脸关键点叠加黑框、金属和圆框眼镜，不重绘五官。
- GFPGAN 后处理：对既有人像生成结果做人脸修复和清晰化，只作为可选后处理。
- 项目成果展示：读取已有 DCGAN、StyleGAN3、InstantID 展示素材。

StyleGAN2-ADA 人脸反演 + 潜空间编辑已取消作为当前交付模块：短步数反演身份偏差明显，官方 projector 额外依赖下载不稳定。StyleGAN3 仍保留为官方 FFHQ 预训练 GAN 上限展示。

## 6. InstantID 前置条件

造型与姿态功能需要 AutoDL 上已有：

```text
external/InstantID/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors
external/InstantID/checkpoints/ip-adapter.bin
external/InstantID/models/antelopev2/
```

还需要 SDXL 基础模型目录。程序会按顺序查找：

```text
INSTANTID_BASE_MODEL_DIR
/root/autodl-fs/models/YamerMIX_v8
/autodl-fs/data/models/YamerMIX_v8
/root/autodl-fs/data/models/YamerMIX_v8
```

如果基础模型在其他位置，启动前设置：

```bash
export INSTANTID_BASE_MODEL_DIR=/你的/YamerMIX_v8/路径
bash scripts/run_facegan_studio.sh
```
