from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}


@dataclass(frozen=True)
class OutputCheck:
    name: str
    directory: Path
    minimum_count: int
    suffixes: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SOTA 与增强模块输出结果")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="项目根目录")
    parser.add_argument("--require-video", action="store_true", help="要求 StyleGAN3 视频存在")
    return parser.parse_args()


def count_media(directory: Path, suffixes: set[str]) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.suffix.lower() in suffixes)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checks = [
        OutputCheck("StyleGAN3 生成人像", root / "outputs" / "stylegan3" / "images", 1, MEDIA_SUFFIXES),
        OutputCheck("AnimeGANv2 动漫化", root / "outputs" / "animegan2", 1, MEDIA_SUFFIXES),
        OutputCheck("CycleGAN 风格迁移", root / "outputs" / "cyclegan_style", 1, MEDIA_SUFFIXES),
        OutputCheck("报告素材", root / "report" / "report_assets", 1, MEDIA_SUFFIXES),
    ]
    if args.require_video:
        checks.append(OutputCheck("StyleGAN3 插值视频", root / "outputs" / "stylegan3", 1, {".mp4"}))

    failed = False
    for check in checks:
        count = count_media(check.directory, check.suffixes)
        ok = count >= check.minimum_count
        status = "通过" if ok else "失败"
        print(f"[{status}] {check.name}: {count} 个文件，目录 {check.directory}")
        failed = failed or not ok

    if failed:
        raise SystemExit("输出结果不完整，请查看上方失败项。")
    print("SOTA 与增强模块输出检查通过。")


if __name__ == "__main__":
    main()
