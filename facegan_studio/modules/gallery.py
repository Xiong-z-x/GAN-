from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .image_utils import collect_images


@dataclass(frozen=True)
class ShowcaseAsset:
    name: str
    path: Path
    description: str


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_showcase_assets(project_root: str | Path) -> list[ShowcaseAsset]:
    root = Path(project_root)
    new_assets = root / "GAN_new_showcase_results" / "report" / "report_assets"
    handoff_assets = root / "docs" / "handoff_assets"
    old_assets = root / "GAN_results_images" / "final_images_package" / "report" / "selected_figures"
    report_assets = root / "report" / "report_assets"

    known: list[tuple[str, list[Path], str]] = [
        (
            "DCGAN 演化展示",
            [
                new_assets / "gan_showcase" / "dcgan_evolution_grid.png",
                report_assets / "gan_showcase" / "dcgan_evolution_grid.png",
                handoff_assets / "dcgan_evolution_grid.png",
            ],
            "展示手写 DCGAN、稳定化 DCGAN 和 DCGAN++ 的阶段性结果。",
        ),
        (
            "StyleGAN3 高质量人像",
            [
                new_assets / "gan_showcase" / "stylegan3_top16_grid.png",
                new_assets / "gan_showcase" / "stylegan3_top32_grid.png",
                handoff_assets / "stylegan3_top16_grid.png",
                handoff_assets / "stylegan3_top32_grid.png",
                old_assets / "fig03_stylegan3_samples.png",
            ],
            "展示成熟 GAN 在 FFHQ 人脸域上的质量上限。",
        ),
        (
            "InstantID 姿态参考",
            [
                new_assets / "instantid_myface_pose" / "pose_reference_grid_4x4.png",
                report_assets / "instantid_myface_pose" / "pose_reference_grid_4x4.png",
                handoff_assets / "pose_reference_grid_4x4.png",
            ],
            "展示个人照片、StyleGAN3 图和公开数据人像构成的姿态参考。",
        ),
        (
            "InstantID 身份保持结果",
            [
                new_assets / "instantid_myface_pose" / "my_face_pose_grid_4x4.png",
                report_assets / "instantid_myface_pose" / "my_face_pose_grid_4x4.png",
                handoff_assets / "my_face_pose_grid_4x4.png",
            ],
            "展示固定个人身份向量后的不同姿态和场景生成。",
        ),
        (
            "AnimeGANv2 多风格",
            [
                old_assets / "fig04_animegan2_comparison.png",
            ],
            "展示 AnimeGANv2 的三种动漫化风格。",
        ),
        (
            "CycleGAN Van Gogh",
            [
                old_assets / "fig05_cyclegan_vangogh_comparison.png",
            ],
            "展示 CycleGAN 艺术风格迁移结果。",
        ),
    ]

    assets: list[ShowcaseAsset] = []
    for name, candidates, description in known:
        path = _first_existing(candidates)
        if path:
            assets.append(ShowcaseAsset(name=name, path=path, description=description))

    if not assets:
        for image_path in collect_images(handoff_assets, recursive=False, limit=12):
            assets.append(ShowcaseAsset(name=image_path.stem, path=image_path, description="迁移交接保留的代表性展示素材。"))

    if not assets:
        for image_path in collect_images(new_assets, recursive=True, limit=12):
            assets.append(ShowcaseAsset(name=image_path.stem, path=image_path, description="项目已有展示素材。"))

    return assets
