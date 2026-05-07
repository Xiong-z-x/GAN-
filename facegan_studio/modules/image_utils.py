from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(path: str | Path, recursive: bool = True, limit: int | None = None) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []

    files: list[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for item in iterator:
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(item)

    files = sorted(files)
    return files[:limit] if limit is not None else files


def load_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def save_rgb(image: Image.Image, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out)
    return out


def center_crop_to_aspect(image: Image.Image, aspect: float) -> Image.Image:
    width, height = image.size
    current = width / height

    if current > aspect:
        new_width = int(height * aspect)
        left = max((width - new_width) // 2, 0)
        return image.crop((left, 0, left + new_width, height))

    new_height = int(width / aspect)
    top = max((height - new_height) // 2, 0)
    return image.crop((0, top, width, top + new_height))


def fit_to_cell(image: Image.Image, cell_size: tuple[int, int], background: str | tuple[int, int, int] = "white") -> Image.Image:
    cell_w, cell_h = cell_size
    fitted = ImageOps.contain(image.convert("RGB"), cell_size)
    canvas = Image.new("RGB", cell_size, background)
    x = (cell_w - fitted.width) // 2
    y = (cell_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def make_grid(
    images: Sequence[Image.Image],
    columns: int = 4,
    cell_size: tuple[int, int] = (256, 256),
    padding: int = 8,
    background: str | tuple[int, int, int] = "white",
) -> Image.Image:
    if not images:
        raise ValueError("没有可用于拼图的图片。")

    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    width = columns * cell_size[0] + (columns - 1) * padding
    height = rows * cell_size[1] + (rows - 1) * padding
    grid = Image.new("RGB", (width, height), background)

    for index, image in enumerate(images):
        cell = fit_to_cell(image, cell_size, background)
        x = (index % columns) * (cell_size[0] + padding)
        y = (index // columns) * (cell_size[1] + padding)
        grid.paste(cell, (x, y))

    return grid


def make_grid_from_paths(
    paths: Sequence[str | Path],
    output_path: str | Path,
    columns: int = 4,
    cell_size: tuple[int, int] = (256, 256),
    padding: int = 8,
) -> Path:
    images = [load_rgb(path) for path in paths]
    grid = make_grid(images, columns=columns, cell_size=cell_size, padding=padding)
    return save_rgb(grid, output_path)


def copy_files(paths: Iterable[str | Path], dest_dir: str | Path) -> list[Path]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in paths:
        src = Path(source)
        if src.exists() and src.is_file():
            out = dest / src.name
            index = 1
            while out.exists():
                out = dest / f"{src.stem}_{index}{src.suffix}"
                index += 1
            shutil.copy2(src, out)
            copied.append(out)
    return copied


def write_metadata(path: str | Path, data: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
