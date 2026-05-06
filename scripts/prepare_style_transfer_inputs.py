from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备动漫化和风格迁移输入图片")
    parser.add_argument("--personal-dir", type=Path, default=Path("data/raw/my_photos"), help="个人照片目录")
    parser.add_argument("--generated-dir", type=Path, default=Path("outputs/stylegan3/images"), help="StyleGAN3 生成图目录")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/style_transfer_inputs"), help="输出目录")
    parser.add_argument("--max-images", type=int, default=8, help="最多整理的图片数量")
    parser.add_argument("--max-side", type=int, default=1024, help="最长边限制")
    parser.add_argument("--clear", action="store_true", help="整理前清空输出目录中的图片")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def resize_keep_ratio(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def clear_images(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        clear_images(args.output_dir)

    personal_images = list_images(args.personal_dir)
    generated_images = list_images(args.generated_dir)
    source_images = [*personal_images, *generated_images]
    if personal_images and generated_images:
        source_name = "个人照片 + StyleGAN3 生成图"
    elif personal_images:
        source_name = "个人照片"
    else:
        source_name = "StyleGAN3 生成图"

    if not source_images:
        raise FileNotFoundError(
            f"未找到输入图片。请放入个人照片到 {args.personal_dir}，或先运行 StyleGAN3 生成图。"
        )

    # 个人照片优先；数量不足时继续补充 StyleGAN3 生成图，避免只有少量个人照片导致展示不足。
    selected_images = source_images[: args.max_images]
    for index, image_path in enumerate(tqdm(selected_images, desc=f"整理{source_name}")):
        target_path = args.output_dir / f"style_input_{index:03d}.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = resize_keep_ratio(image, args.max_side)
            image.save(target_path)

    # 备份首张示例，方便报告里说明输入来源。
    backup_dir = args.output_dir.parent / "style_transfer_input_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_images[0], backup_dir / selected_images[0].name)
    print(f"已使用{source_name}准备输入图片：{len(selected_images)}")


if __name__ == "__main__":
    main()
