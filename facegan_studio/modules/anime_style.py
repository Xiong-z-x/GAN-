from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .image_utils import collect_images, copy_files, load_rgb, make_grid, save_rgb, write_metadata
from .paths import ProjectPaths, current_timestamp


ANIME_STYLES = {
    "face_paint_512_v2": "Face Paint v2",
    "face_paint_512_v1": "Face Paint v1",
    "paprika": "Paprika",
}

CYCLEGAN_STYLES = {
    "style_vangogh": "Van Gogh",
    "style_monet": "Monet",
    "style_ukiyoe": "Ukiyoe",
}


@dataclass(frozen=True)
class AnimeStyleResult:
    output_dir: Path
    image_paths: list[Path]
    grid_path: Path
    metadata_path: Path


def _require_bash_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"未找到脚本：{script_path}")


def _prepare_single_input(image_path: Path, input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    for old_file in input_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()
    suffix = image_path.suffix.lower() if image_path.suffix else ".png"
    target = input_dir / f"input_face{suffix}"
    shutil.copy2(image_path, target)
    return target


def run_anime_styles(
    image_path: str | Path,
    project_root: str | Path | None = None,
    styles: list[str] | None = None,
    include_cyclegan: bool = False,
    timestamp: str | None = None,
) -> AnimeStyleResult:
    paths = ProjectPaths(project_root)
    stamp = timestamp or current_timestamp()
    run_dir = paths.create_run_dir("anime", stamp)
    report_dir = paths.create_report_run_dir("anime", stamp)
    input_dir = run_dir / "input"
    input_file = _prepare_single_input(Path(image_path), input_dir)

    script = paths.project_root / "scripts" / "run_animegan2_infer.sh"
    _require_bash_script(script)

    selected = styles or list(ANIME_STYLES)
    result_paths: list[Path] = []
    subprocess_logs: dict[str, str] = {}

    for style in selected:
        if style not in ANIME_STYLES:
            continue
        out_dir = run_dir / style
        env = os.environ.copy()
        env.update(
            {
                "ANIMEGAN2_INPUT_DIR": str(input_dir),
                "ANIMEGAN2_OUTDIR": str(out_dir),
                "ANIMEGAN2_STYLE": style,
            }
        )
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=str(paths.project_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        subprocess_logs[style] = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"AnimeGANv2 风格 {style} 运行失败：\n{subprocess_logs[style]}")
        style_images = collect_images(out_dir, recursive=False)
        result_paths.extend(style_images)

    if include_cyclegan:
        cyclegan_script = paths.project_root / "scripts" / "run_cyclegan_pretrained_style.sh"
        _require_bash_script(cyclegan_script)
        for style in ("style_vangogh",):
            out_dir = run_dir / style
            env = os.environ.copy()
            env.update(
                {
                    "CYCLEGAN_INPUT_DIR": str(input_dir),
                    "CYCLEGAN_RESULTS_DIR": str(out_dir),
                    "CYCLEGAN_MODEL_NAME": style,
                    "CYCLEGAN_NUM_TEST": "1",
                }
            )
            completed = subprocess.run(
                ["bash", str(cyclegan_script)],
                cwd=str(paths.project_root),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            subprocess_logs[style] = completed.stdout + completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(f"CycleGAN 风格 {style} 运行失败：\n{subprocess_logs[style]}")
            fake_images = [path for path in collect_images(out_dir, recursive=True) if "fake" in path.stem]
            result_paths.extend(fake_images)

    if not result_paths:
        raise RuntimeError("动漫风格化没有生成任何图片。")

    grid = make_grid([load_rgb(path) for path in result_paths], columns=min(4, len(result_paths)), cell_size=(256, 256))
    grid_path = save_rgb(grid, run_dir / "anime_grid.png")

    copied = copy_files([input_file, *result_paths, grid_path], report_dir)
    metadata_path = write_metadata(
        run_dir / "metadata.json",
        {
            "mode": "anime",
            "styles": selected,
            "include_cyclegan": include_cyclegan,
            "input": str(input_file),
            "outputs": [str(path) for path in result_paths],
            "grid": str(grid_path),
            "report_assets": [str(path) for path in copied],
            "logs": subprocess_logs,
        },
    )
    shutil.copy2(metadata_path, report_dir / "metadata.json")

    return AnimeStyleResult(output_dir=run_dir, image_paths=result_paths, grid_path=grid_path, metadata_path=metadata_path)
