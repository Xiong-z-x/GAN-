from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps

from .image_utils import collect_images, load_rgb, make_grid, save_rgb, write_metadata
from .paths import ProjectPaths, current_timestamp


ACCESSORY_STYLES = ["original", "black_glasses", "metal_glasses", "round_glasses"]


@dataclass(frozen=True)
class IdentityAccessoryResult:
    output_dir: Path
    image_paths: list[Path]
    grid_path: Path
    metadata_path: Path


def draw_glasses_overlay(
    image: Image.Image,
    keypoints: Sequence[tuple[float, float]],
    style: str = "black_glasses",
) -> Image.Image:
    if len(keypoints) < 2:
        raise ValueError("眼镜叠加至少需要左右眼关键点。")

    left_eye = keypoints[0]
    right_eye = keypoints[1]
    eye_dist = max(((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2) ** 0.5, 1.0)
    lens_w = eye_dist * (0.72 if style == "round_glasses" else 0.66)
    lens_h = lens_w * (0.72 if style == "round_glasses" else 0.58)
    line_w = max(2, int(eye_dist * (0.055 if style == "black_glasses" else 0.035)))
    radius = int(lens_h * (0.45 if style == "round_glasses" else 0.22))
    color = (18, 18, 18, 230) if style in {"black_glasses", "round_glasses"} else (120, 126, 130, 220)

    out = ImageOps.exif_transpose(image).convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def box(center: tuple[float, float]) -> tuple[float, float, float, float]:
        return (
            center[0] - lens_w / 2,
            center[1] - lens_h / 2,
            center[0] + lens_w / 2,
            center[1] + lens_h / 2,
        )

    left_box = box(left_eye)
    right_box = box(right_eye)
    draw.rounded_rectangle(left_box, radius=radius, outline=color, width=line_w)
    draw.rounded_rectangle(right_box, radius=radius, outline=color, width=line_w)

    bridge_y = (left_eye[1] + right_eye[1]) / 2
    draw.line(
        [(left_box[2], bridge_y), (right_box[0], bridge_y)],
        fill=color,
        width=max(1, line_w - 1),
    )
    temple_len = eye_dist * 0.45
    draw.line(
        [(left_box[0], bridge_y), (left_box[0] - temple_len, bridge_y - lens_h * 0.18)],
        fill=color,
        width=max(1, line_w - 1),
    )
    draw.line(
        [(right_box[2], bridge_y), (right_box[2] + temple_len, bridge_y - lens_h * 0.18)],
        fill=color,
        width=max(1, line_w - 1),
    )

    return Image.alpha_composite(out, overlay).convert("RGB")


def _detect_keypoints(image_path: Path, project_root: Path) -> list[tuple[float, float]]:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    instantid_root = project_root / "external" / "InstantID"
    app = FaceAnalysis(
        name="antelopev2",
        root=str(instantid_root),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = load_rgb(image_path)
    faces = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    if not faces:
        raise RuntimeError(f"未检测到人脸，无法叠加眼镜：{image_path}")
    face = max(faces, key=lambda item: float((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])))
    return [(float(x), float(y)) for x, y in face["kps"]]


def run_identity_accessory_styles(
    project_root: str | Path,
    input_paths: Sequence[str | Path] | None = None,
    timestamp: str | None = None,
    sync_handoff: bool = False,
) -> IdentityAccessoryResult:
    paths = ProjectPaths(project_root)
    photos = [Path(path) for path in input_paths] if input_paths else collect_images(paths.project_root / "data" / "raw" / "my_photos", recursive=True, limit=4)
    if len(photos) < 4:
        raise FileNotFoundError(f"需要 4 张本人照片，当前只找到 {len(photos)} 张。")

    run_dir = paths.create_run_dir("identity_accessory", timestamp or current_timestamp())
    report_dir = paths.create_report_run_dir("identity_accessory", run_dir.name)
    output_paths: list[Path] = []
    metadata_items: list[dict[str, str]] = []

    for source_index, source in enumerate(photos[:4]):
        image = load_rgb(source)
        keypoints = _detect_keypoints(source, paths.project_root)
        for style in ACCESSORY_STYLES:
            styled = image if style == "original" else draw_glasses_overlay(image, keypoints, style=style)
            out_path = save_rgb(styled, run_dir / f"identity_accessory_p{source_index:02d}_{style}.png")
            shutil.copy2(out_path, report_dir / out_path.name)
            output_paths.append(out_path)
            metadata_items.append({"style": style, "source": str(source), "output": str(out_path)})

    grid = make_grid([load_rgb(path) for path in output_paths], columns=4, cell_size=(256, 256), padding=0)
    grid_path = save_rgb(grid, run_dir / "identity_accessory_grid.png")
    shutil.copy2(grid_path, report_dir / grid_path.name)

    metadata_path = write_metadata(
        run_dir / "metadata.json",
        {
            "mode": "identity_preserving_accessory_styles",
            "items": metadata_items,
            "note": "为满足人脸不失真要求，本模块只按人脸关键点叠加眼镜，不重绘五官、不改变脸型；四张本人照片均作为输入。",
        },
    )
    shutil.copy2(metadata_path, report_dir / "metadata.json")

    if sync_handoff:
        # Do not copy this grid into docs/handoff_assets: it contains personal photos.
        # The report copy under report/report_assets is ignored by Git and remains local.
        pass

    return IdentityAccessoryResult(run_dir, output_paths, grid_path, metadata_path)
