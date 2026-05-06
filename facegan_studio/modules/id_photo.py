from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from .image_utils import center_crop_to_aspect, load_rgb, make_grid, save_rgb, write_metadata


ID_BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "blue": (67, 142, 219),
    "red": (210, 48, 48),
}

BACKGROUND_LABELS: dict[str, str] = {
    "white": "白底",
    "blue": "蓝底",
    "red": "红底",
}


@dataclass(frozen=True)
class IdPhotoResult:
    output_dir: Path
    variants: dict[str, Path]
    grid_path: Path
    metadata_path: Path


def _enhance_portrait(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.04)
    image = ImageEnhance.Contrast(image).enhance(1.05)
    image = ImageEnhance.Color(image).enhance(1.02)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))


def _make_conservative_id_photo(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    target_w, target_h = size
    aspect = target_w / target_h
    crop = center_crop_to_aspect(image, aspect)
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    crop = _enhance_portrait(crop)

    canvas = Image.new("RGB", size, background)
    # 保守融合：不做强分割，保留原始人脸细节，只替换边缘背景感。
    inner_margin_x = max(6, target_w // 28)
    inner_margin_y = max(6, target_h // 28)
    inner = crop.crop((inner_margin_x, inner_margin_y, target_w - inner_margin_x, target_h - inner_margin_y))
    canvas.paste(inner, (inner_margin_x, inner_margin_y))
    return canvas


def create_id_photo_variants(
    image_path: str | Path,
    output_dir: str | Path,
    backgrounds: list[str] | None = None,
    size: tuple[int, int] = (413, 626),
) -> IdPhotoResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path)
    selected = backgrounds or ["white", "blue", "red"]
    variants: dict[str, Path] = {}

    input_copy = save_rgb(image, out_dir / "input.png")

    for key in selected:
        if key not in ID_BACKGROUNDS:
            continue
        result = _make_conservative_id_photo(image, size=size, background=ID_BACKGROUNDS[key])
        out_path = out_dir / f"id_photo_{key}.png"
        save_rgb(result, out_path)
        variants[key] = out_path

    if not variants:
        raise ValueError("没有有效的证件照背景选项。")

    grid_images = [load_rgb(path) for path in variants.values()]
    grid = make_grid(grid_images, columns=len(grid_images), cell_size=(220, 320), padding=10)
    grid_path = save_rgb(grid, out_dir / "id_photo_grid.png")

    metadata_path = write_metadata(
        out_dir / "metadata.json",
        {
            "mode": "id_photo",
            "input": str(input_copy),
            "size": size,
            "backgrounds": {key: BACKGROUND_LABELS.get(key, key) for key in variants},
            "variants": {key: str(path) for key, path in variants.items()},
            "note": "证件照模块采用保守裁剪、背景融合和轻度增强，不作为正式证件照审核保证。",
        },
    )

    return IdPhotoResult(output_dir=out_dir, variants=variants, grid_path=grid_path, metadata_path=metadata_path)
