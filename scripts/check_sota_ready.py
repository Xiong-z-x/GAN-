from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SOTA 与增强模块运行条件")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="项目根目录")
    parser.add_argument("--strict-input", action="store_true", help="要求风格迁移输入目录已有图片")
    return parser.parse_args()


def check_file(name: str, path: Path) -> CheckResult:
    return CheckResult(name=name, ok=path.is_file(), detail=str(path))


def check_dir(name: str, path: Path) -> CheckResult:
    return CheckResult(name=name, ok=path.is_dir(), detail=str(path))


def has_images(path: Path) -> bool:
    return path.exists() and any(item.suffix.lower() in IMAGE_SUFFIXES for item in path.rglob("*"))


def check_python_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    packages = ["torch", "torchvision", "PIL", "numpy", "scipy", "tqdm", "click", "imageio"]
    for package in packages:
        try:
            __import__(package)
        except Exception as exc:
            results.append(CheckResult(package, False, str(exc)))
        else:
            results.append(CheckResult(package, True, "已安装"))

    try:
        import torch
    except Exception as exc:
        results.append(CheckResult("CUDA", False, f"无法导入 PyTorch：{exc}"))
    else:
        detail = "可用" if torch.cuda.is_available() else "不可用"
        results.append(CheckResult("CUDA", torch.cuda.is_available(), detail))
    return results


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    external = root / "external"
    results: list[CheckResult] = []

    results.extend(check_python_packages())
    results.append(CheckResult("git", shutil.which("git") is not None, shutil.which("git") or "未找到"))
    results.append(CheckResult("bash", shutil.which("bash") is not None, shutil.which("bash") or "未找到"))
    results.append(CheckResult("wget", shutil.which("wget") is not None, shutil.which("wget") or "未找到"))
    results.append(CheckResult("nvcc", shutil.which("nvcc") is not None, shutil.which("nvcc") or "未找到，StyleGAN 自定义算子可能无法编译"))

    results.extend(
        [
            check_file("StyleGAN3 生成入口", external / "stylegan3" / "gen_images.py"),
            check_file("StyleGAN3 视频入口", external / "stylegan3" / "gen_video.py"),
            check_file("AnimeGANv2 推理入口", external / "animegan2-pytorch" / "test.py"),
            check_file("AnimeGANv2 默认权重", external / "animegan2-pytorch" / "weights" / "celeba_distill.pt"),
            check_file("CycleGAN 推理入口", external / "pytorch-CycleGAN-and-pix2pix" / "test.py"),
            check_file("CycleGAN 权重下载脚本", external / "pytorch-CycleGAN-and-pix2pix" / "scripts" / "download_cyclegan_model.sh"),
            check_file("StyleGAN2-ADA projector", external / "stylegan2-ada-pytorch" / "projector.py"),
            check_dir("报告素材目录", root / "report" / "report_assets"),
        ]
    )

    input_dir = root / "data" / "processed" / "style_transfer_inputs"
    generated_dir = root / "outputs" / "stylegan3" / "images"
    personal_dir = root / "data" / "raw" / "my_photos"
    if args.strict_input:
        input_ok = has_images(input_dir)
        input_detail = str(input_dir)
    else:
        input_ok = True
        has_any_input = has_images(input_dir) or has_images(generated_dir) or has_images(personal_dir)
        input_detail = "已有可用输入" if has_any_input else "非严格模式，流水线会先生成 StyleGAN3 图片再准备输入"
    results.append(CheckResult("增强模块输入图片", input_ok, input_detail))

    failed = [result for result in results if not result.ok]
    for result in results:
        status = "通过" if result.ok else "失败"
        print(f"[{status}] {result.name}: {result.detail}")

    if failed:
        print("\n存在未满足条件。若外部仓库缺失，请先运行 scripts/setup_external_repos.sh。")
        sys.exit(1)

    print("\nSOTA 与增强模块运行条件检查通过。")


if __name__ == "__main__":
    main()
