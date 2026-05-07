# AutoDL 新工作区启动指令

用途：当新对话直接运行在 AutoDL 上，而 AutoDL 当前还没有本仓库文档和代码时，先用本文把工作区建立起来。不要先要求阅读仓库内 md，因为那些文件此时还不存在。

## 1. 首选：从 GitHub 拉取

```bash
cd /root/autodl-tmp
if [ -d GAN/.git ]; then
  cd GAN
  git status --short --branch
  git pull --ff-only
else
  git clone https://github.com/Xiong-z-x/GAN-.git GAN
  cd GAN
fi

sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh

echo "当前提交："
git log -1 --oneline
echo "当前状态："
git status --short --branch
```

如果 `git pull --ff-only` 失败，不要执行 `git reset --hard`。先查看：

```bash
git status --short
```

判断 AutoDL 本地是否存在需要保留的代码改动。

## 2. 兜底：从本地 D 盘或 C 盘复制轻量源码

只在 GitHub 暂时不可用时使用。下面命令在本地 Windows PowerShell 执行，不是在 AutoDL shell 里执行。

```powershell
$LOCAL = "D:\GAN"
if (-not (Test-Path $LOCAL)) {
  $LOCAL = "C:\GAN"
}

ssh autodl-gan "mkdir -p /root/autodl-tmp/GAN"
scp -r `
  "$LOCAL\README.md" `
  "$LOCAL\AutoDL_运行指南.md" `
  "$LOCAL\FaceGAN_Studio_运行说明.md" `
  "$LOCAL\requirements_autodl.txt" `
  "$LOCAL\代码附录.md" `
  "$LOCAL\文件说明.md" `
  "$LOCAL\报告素材使用说明.md" `
  "$LOCAL\最终提交清单.md" `
  "$LOCAL\docs" `
  "$LOCAL\facegan_studio" `
  "$LOCAL\scripts" `
  "$LOCAL\src" `
  "$LOCAL\tests" `
  autodl-gan:/root/autodl-tmp/GAN/
```

不要默认复制这些目录：

```text
data/
outputs/
external/
GAN_results_images/
GAN_new_showcase_results/
report/report_assets/
```

原因：

- 这些目录体积大，或包含数据、个人照片、权重、缓存、外部仓库。
- AutoDL 端应按需要重新准备或复用已有目录。
- 如果确实要迁移完整结果包，必须单独打包并明确目标目录。

## 3. 建立工作区后必须阅读

```bash
cd /root/autodl-tmp/GAN
sed -n '1,220p' README.md
sed -n '1,260p' docs/handoff/迁移前交接总报告.md
sed -n '1,260p' docs/handoff/后续模型注意事项.md
sed -n '1,260p' AutoDL_运行指南.md
```

## 4. 最小验证

```bash
cd /root/autodl-tmp/GAN
python -m compileall src scripts facegan_studio
python -m facegan_studio.app --help
```

之后再进入 FaceGAN Studio、AnimeGANv2/CycleGAN、InstantID 等重模型验证。
