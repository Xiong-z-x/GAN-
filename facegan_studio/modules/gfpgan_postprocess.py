from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .image_utils import collect_images, load_rgb, make_grid, save_rgb, write_metadata
from .paths import ProjectPaths


@dataclass(frozen=True)
class GFPGANResult:
    output_dir: Path
    enhanced_paths: list[Path]
    comparison_grid: Path
    metadata_path: Path


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.name, path.stat().st_mtime), reverse=True)[0]


def select_default_gfpgan_inputs(project_root: str | Path, limit: int = 8) -> list[Path]:
    """Select representative already-generated portraits for GFPGAN enhancement."""
    paths = ProjectPaths(project_root)
    selected: list[Path] = []

    latest_pose = _latest_run_dir(paths.outputs_dir / "pose_style")
    if latest_pose:
        selected.extend(sorted(latest_pose.glob("pose_style_*.png")))

    latest_anime = _latest_run_dir(paths.outputs_dir / "anime")
    if latest_anime:
        selected.extend(path for path in collect_images(latest_anime, recursive=True) if path.name != "input_face.png")
        grid = latest_anime / "anime_grid.png"
        if grid.exists():
            selected.append(grid)

    handoff = paths.project_root / "docs" / "handoff_assets"
    for name in ["stylegan3_top16_grid.png", "my_face_pose_grid_4x4.png", "dcgan_evolution_grid.png"]:
        path = handoff / name
        if path.exists():
            selected.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            unique.append(path)
    return unique[:limit]


def build_gfpgan_comparison_grid(
    pairs: Sequence[tuple[str | Path, str | Path]],
    output_path: str | Path,
    cell_size: tuple[int, int] = (256, 256),
) -> Path:
    images = []
    for original, enhanced in pairs:
        images.append(load_rgb(original))
        images.append(load_rgb(enhanced))
    grid = make_grid(images, columns=2, cell_size=cell_size, padding=10)
    return save_rgb(grid, output_path)


def _copy_or_enhance_fallback(input_path: Path, output_path: Path) -> None:
    """Keep pipeline usable in tests or dependency-free dry runs."""
    save_rgb(load_rgb(input_path), output_path)


def _create_gfpganer(model_path: str | Path | None, upscale: int):
    try:
        from gfpgan import GFPGANer
    except Exception as error:  # pragma: no cover - covered by integration command.
        raise RuntimeError(
            "未安装 GFPGAN 依赖。请先运行：python -m pip install gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0"
        ) from error

    model = str(model_path) if model_path else "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    return GFPGANer(
        model_path=model,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def run_gfpgan_postprocess(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    model_path: str | Path | None = None,
    upscale: int = 1,
    suffix: str = "_gfpgan",
    dry_run: bool = False,
) -> GFPGANResult:
    out_dir = Path(output_dir)
    enhanced_dir = out_dir / "enhanced"
    original_dir = out_dir / "original"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in inputs if Path(path).exists()]
    if not input_paths:
        raise ValueError("没有可用于 GFPGAN 后处理的输入图片。")

    gfpganer = None if dry_run else _create_gfpganer(model_path, upscale)
    enhanced_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []

    for index, input_path in enumerate(input_paths):
        original_copy = original_dir / f"{index:02d}_{input_path.name}"
        shutil.copy2(input_path, original_copy)
        enhanced_path = enhanced_dir / f"{index:02d}_{input_path.stem}{suffix}.png"

        if dry_run:
            _copy_or_enhance_fallback(input_path, enhanced_path)
        else:
            import cv2

            image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片：{input_path}")
            _, _, restored = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
            cv2.imwrite(str(enhanced_path), restored)

        enhanced_paths.append(enhanced_path)
        pairs.append((original_copy, enhanced_path))

    grid_path = build_gfpgan_comparison_grid(pairs, out_dir / "gfpgan_comparison_grid.png")
    metadata = {
        "mode": "gfpgan_postprocess",
        "note": "GFPGAN 仅作为人脸修复和清晰化后处理；它可能改变局部五官细节，不作为身份保持模型。",
        "inputs": [str(path) for path in input_paths],
        "enhanced": [str(path) for path in enhanced_paths],
        "comparison_grid": str(grid_path),
    }
    metadata_path = write_metadata(out_dir / "metadata.json", metadata)
    return GFPGANResult(out_dir, enhanced_paths, grid_path, metadata_path)


def sync_gfpgan_report_assets(project_root: str | Path, result: GFPGANResult) -> Path:
    paths = ProjectPaths(project_root)
    report_dir = paths.create_report_run_dir("gfpgan_postprocess", result.output_dir.name)
    for path in [*result.enhanced_paths, result.comparison_grid, result.metadata_path]:
        shutil.copy2(path, report_dir / path.name)
    return report_dir
