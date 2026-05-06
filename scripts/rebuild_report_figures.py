from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "GAN_results_images" / "final_images_package"
FIG_DIR = PACKAGE_ROOT / "report" / "selected_figures"


def fit_square(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def make_grid(images: list[Image.Image], rows: int, cols: int, cell_size: int, pad: int = 16) -> Image.Image:
    canvas = Image.new("RGB", (cols * cell_size + (cols + 1) * pad, rows * cell_size + (rows + 1) * pad), (255, 255, 255))
    for idx, image in enumerate(images):
        if idx >= rows * cols:
            break
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_size + pad)
        y = pad + r * (cell_size + pad)
        canvas.paste(fit_square(image, cell_size), (x, y))
    return canvas


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_dcgan_cells(grid_path: Path, indices: list[int], grid_size: int = 8, pad: int = 2) -> list[Image.Image]:
    image = load_rgb(grid_path)
    tile = (image.width - (grid_size + 1) * pad) // grid_size
    cells: list[Image.Image] = []
    for index in indices:
        row, col = divmod(index, grid_size)
        left = pad + col * (tile + pad)
        top = pad + row * (tile + pad)
        cells.append(image.crop((left, top, left + tile, top + tile)))
    return cells


def save_image(image: Image.Image, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    image.save(path)
    return path


def build_dcgan_progress() -> Path:
    # 将多轮训练过程图抽成同一张拼图，便于在报告里展示收敛趋势。
    paths = [
        PACKAGE_ROOT / "outputs" / "dcgan" / "report_progress" / "dcgan_epoch_001.png",
        PACKAGE_ROOT / "outputs" / "dcgan" / "report_progress" / "dcgan_epoch_010.png",
        PACKAGE_ROOT / "outputs" / "dcgan" / "report_progress" / "dcgan_epoch_030.png",
        PACKAGE_ROOT / "outputs" / "dcgan" / "report_progress" / "dcgan_epoch_060.png",
    ]
    indices = [0, 9, 54, 63]
    tiles: list[Image.Image] = []
    for path in paths:
        tiles.extend(crop_dcgan_cells(path, indices))
    return save_image(make_grid(tiles, rows=4, cols=4, cell_size=160, pad=14), "fig01_dcgan_progress.png")


def build_dcgan_samples() -> Path:
    # 最终轮次的 16 张样例单独排版，避免 8x8 过于拥挤。
    path = PACKAGE_ROOT / "outputs" / "dcgan" / "samples" / "epoch_060.png"
    indices = [
        0, 2, 4, 6,
        16, 18, 20, 22,
        32, 34, 36, 38,
        48, 50, 52, 54,
    ]
    tiles = crop_dcgan_cells(path, indices)
    return save_image(make_grid(tiles, rows=4, cols=4, cell_size=160, pad=14), "fig02_dcgan_generated_grids.png")


def build_stylegan3_samples() -> Path:
    # StyleGAN3 结果采用 4x4 代表性抽样，突出高分辨率细节。
    paths = sorted((PACKAGE_ROOT / "outputs" / "stylegan3" / "images").glob("seed*.png"))
    selected = [paths[i] for i in range(min(16, len(paths)))]
    return save_image(make_grid([load_rgb(p) for p in selected], rows=4, cols=4, cell_size=220, pad=16), "fig03_stylegan3_samples.png")


def build_animegan_comparison() -> Path:
    # 动漫化部分采用“原图 + 三种风格”四行对比。
    stems = [0, 9, 18, 27]
    rows: list[Image.Image] = []
    sources = [
        PACKAGE_ROOT / "outputs" / "anime_report_selected" / "original",
        PACKAGE_ROOT / "outputs" / "anime_report_selected" / "face_paint_v2",
        PACKAGE_ROOT / "outputs" / "anime_report_selected" / "face_paint_v1",
        PACKAGE_ROOT / "outputs" / "anime_report_selected" / "paprika",
    ]
    for stem in stems:
        name = f"anime_input_{stem:03d}.png"
        for src in sources:
            rows.append(load_rgb(src / name))
    return save_image(make_grid(rows, rows=4, cols=4, cell_size=220, pad=16), "fig04_animegan2_comparison.png")


def build_cyclegan_style(style_dir: Path, prefix: str, stems: list[int], out_name: str) -> Path:
    # CycleGAN 结果采用 2x4 布局，左列为输入，右列为输出。
    images: list[Image.Image] = []
    for stem in stems:
        base = f"{prefix}_{stem:03d}"
        images.append(load_rgb(style_dir / f"{base}_real.png"))
        images.append(load_rgb(style_dir / f"{base}_fake.png"))
    return save_image(make_grid(images, rows=2, cols=4, cell_size=240, pad=18), out_name)


def main() -> None:
    outputs = [
        build_dcgan_progress(),
        build_dcgan_samples(),
        build_stylegan3_samples(),
        build_animegan_comparison(),
        build_cyclegan_style(
            PACKAGE_ROOT / "outputs" / "cyclegan_style" / "style_vangogh_pretrained" / "test_latest" / "images",
            "style_input",
            [0, 1, 2, 3],
            "fig05_cyclegan_vangogh_comparison.png",
        ),
        build_cyclegan_style(
            PACKAGE_ROOT / "outputs" / "cyclegan_style_monet" / "style_monet_pretrained" / "test_latest" / "images",
            "style_input",
            [0, 1, 2, 3],
            "fig06_cyclegan_monet_comparison.png",
        ),
        build_cyclegan_style(
            PACKAGE_ROOT / "outputs" / "cyclegan_style_ukiyoe" / "style_ukiyoe_pretrained" / "test_latest" / "images",
            "anime_input",
            [0, 1, 2, 3],
            "fig07_cyclegan_ukiyoe_comparison.png",
        ),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
