from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 CelebA 图像为 DCGAN 训练目录")
    parser.add_argument("--source-dir", type=Path, required=True, help="原始图像目录")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/celeba_64"), help="输出目录")
    parser.add_argument("--image-size", type=int, default=64, help="输出图像尺寸")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张，0 表示不限制")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in args.source_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"未找到图像文件：{args.source_dir}")

    for index, image_path in enumerate(tqdm(image_paths, desc="处理 CelebA")):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            image.save(args.output_dir / f"celeba_{index:06d}.png")


if __name__ == "__main__":
    main()

