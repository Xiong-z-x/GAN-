from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理个人照片为后续 demo 输入")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/my_photos"), help="个人照片原图目录")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/style_transfer_inputs"), help="输出目录")
    parser.add_argument("--max-side", type=int, default=1024, help="最长边限制")
    return parser.parse_args()


def resize_keep_ratio(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in args.source_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"未找到个人照片：{args.source_dir}")

    # 统一把个人照片整理成适合后续推理的 PNG 形式。
    for index, image_path in enumerate(tqdm(image_paths, desc="整理个人照片")):
        target_path = args.output_dir / f"personal_{index:03d}.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = resize_keep_ratio(image, args.max_side)
            image.save(target_path)

    # 保留一份原始示例备份，避免后续输入整理时误删源图。
    backup_dir = args.output_dir / "原始示例备份"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_paths[0], backup_dir / image_paths[0].name)


if __name__ == "__main__":
    main()
