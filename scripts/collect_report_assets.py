from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收集报告使用的图片和视频素材")
    parser.add_argument("--output-dir", type=Path, default=Path("report/report_assets"), help="报告素材目录")
    parser.add_argument(
        "--sources",
        nargs="*",
        type=Path,
        default=[
            Path("outputs/dcgan"),
            Path("outputs/stylegan3"),
            Path("outputs/animegan2"),
            Path("outputs/cyclegan_style"),
            Path("outputs/stylegan2ada_projector_optional"),
        ],
        help="需要扫描的输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for source in args.sources:
        if not source.exists():
            continue
        for media_path in source.rglob("*"):
            if media_path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            # 用源目录名做前缀，避免不同模块导出的文件互相覆盖。
            relative_name = "_".join(media_path.relative_to(source).parts)
            target_path = args.output_dir / f"{source.name}_{relative_name}"
            shutil.copy2(media_path, target_path)
            copied += 1

    print(f"已收集素材数量：{copied}")


if __name__ == "__main__":
    main()
