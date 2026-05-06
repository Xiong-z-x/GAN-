from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import AppConfig
from .modules.anime_style import ANIME_STYLES, run_anime_styles
from .modules.face_detector import detect_face_preview
from .modules.gallery import collect_showcase_assets
from .modules.id_photo import BACKGROUND_LABELS, create_id_photo_variants
from .modules.paths import ProjectPaths
from .modules.pose_styler import generate_pose_styles


def _require_image(image_path: str | None) -> Path:
    if not image_path:
        raise ValueError("请先上传一张人脸图片。")
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"上传图片不存在：{path}")
    return path


def _gallery_items(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _format_error(error: Exception) -> str:
    return f"运行失败：{type(error).__name__}: {error}"


def build_app(config: AppConfig) -> Any:
    import gradio as gr

    project_paths = ProjectPaths(config.project_root)

    def preview_face(image_path: str | None) -> tuple[str | None, str]:
        try:
            image = _require_image(image_path)
            run_dir = project_paths.create_run_dir("preview")
            result = detect_face_preview(image, run_dir)
            return str(result.preview_path), result.message
        except Exception as error:
            return None, _format_error(error)

    def run_anime(
        image_path: str | None,
        style_labels: list[str],
        include_cyclegan: bool,
    ) -> tuple[list[str], str | None, str]:
        try:
            image = _require_image(image_path)
            label_to_key = {label: key for key, label in ANIME_STYLES.items()}
            selected = [label_to_key[label] for label in style_labels if label in label_to_key]
            if not selected:
                selected = list(ANIME_STYLES)
            result = run_anime_styles(
                image,
                project_root=config.project_root,
                styles=selected,
                include_cyclegan=include_cyclegan,
            )
            return _gallery_items(result.image_paths), str(result.grid_path), f"完成。输出目录：{result.output_dir}"
        except Exception as error:
            return [], None, _format_error(error)

    def run_id_photo(
        image_path: str | None,
        background_labels: list[str],
        size_label: str,
    ) -> tuple[list[str], str | None, str]:
        try:
            image = _require_image(image_path)
            label_to_key = {label: key for key, label in BACKGROUND_LABELS.items()}
            backgrounds = [label_to_key[label] for label in background_labels if label in label_to_key]
            if not backgrounds:
                backgrounds = ["white", "blue", "red"]

            size_map = {
                "一寸 413x626": (413, 626),
                "二寸 626x413": (626, 413),
                "方形头像 512x512": (512, 512),
            }
            size = size_map.get(size_label, (413, 626))
            run_dir = project_paths.create_run_dir("id_photo")
            result = create_id_photo_variants(image, run_dir, backgrounds=backgrounds, size=size)
            project_paths.create_report_run_dir("id_photo", run_dir.name)
            report_dir = project_paths.report_assets_dir / "id_photo" / run_dir.name
            report_dir.mkdir(parents=True, exist_ok=True)
            for path in [*result.variants.values(), result.grid_path, result.metadata_path]:
                target = report_dir / path.name
                target.write_bytes(path.read_bytes())
            return _gallery_items(list(result.variants.values())), str(result.grid_path), f"完成。输出目录：{result.output_dir}"
        except Exception as error:
            return [], None, _format_error(error)

    def run_pose(
        image_path: str | None,
        count: int,
        strength_label: str,
    ) -> tuple[list[str], str | None, str | None, str]:
        try:
            image = _require_image(image_path)
            strength = "strong" if strength_label == "更像本人" else "standard"
            result = generate_pose_styles(
                image,
                project_root=config.project_root,
                count=int(count),
                identity_strength=strength,
            )
            return (
                _gallery_items(result.image_paths),
                str(result.grid_path),
                str(result.reference_grid_path),
                f"完成。输出目录：{result.output_dir}",
            )
        except Exception as error:
            return [], None, None, _format_error(error)

    def load_showcase() -> tuple[list[tuple[str, str]], str]:
        assets = collect_showcase_assets(config.project_root)
        items = [(str(asset.path), asset.name) for asset in assets]
        if not assets:
            return [], "没有找到已有展示素材。"
        lines = [f"{asset.name}: {asset.description}" for asset in assets]
        return items, "\n".join(lines)

    with gr.Blocks(title="FaceGAN Studio") as app:
        gr.Markdown(
            """
            # FaceGAN Studio

            基于 GAN 风格迁移、成熟 GAN 展示和身份保持生成的人脸应用封装。
            上传图片后可以生成动漫风、证件照、眼镜/造型/不同姿态结果。
            """
        )

        with gr.Tab("输入预览"):
            input_image = gr.Image(type="filepath", label="上传人脸图片")
            preview_button = gr.Button("检测人脸")
            preview_image = gr.Image(type="filepath", label="人脸检测预览")
            preview_status = gr.Textbox(label="状态", lines=3)
            preview_button.click(preview_face, inputs=[input_image], outputs=[preview_image, preview_status])

        with gr.Tab("动漫风格化"):
            anime_input = gr.Image(type="filepath", label="上传人脸图片")
            anime_styles = gr.CheckboxGroup(
                choices=list(ANIME_STYLES.values()),
                value=list(ANIME_STYLES.values()),
                label="AnimeGANv2 风格",
            )
            include_cyclegan = gr.Checkbox(value=False, label="追加 CycleGAN Van Gogh 风格")
            anime_button = gr.Button("生成动漫风")
            anime_gallery = gr.Gallery(label="动漫风结果", columns=4, height=520)
            anime_grid = gr.Image(type="filepath", label="动漫风拼图")
            anime_status = gr.Textbox(label="状态", lines=3)
            anime_button.click(
                run_anime,
                inputs=[anime_input, anime_styles, include_cyclegan],
                outputs=[anime_gallery, anime_grid, anime_status],
            )

        with gr.Tab("证件照生成"):
            id_input = gr.Image(type="filepath", label="上传人脸图片")
            id_backgrounds = gr.CheckboxGroup(
                choices=list(BACKGROUND_LABELS.values()),
                value=list(BACKGROUND_LABELS.values()),
                label="背景颜色",
            )
            id_size = gr.Dropdown(
                choices=["一寸 413x626", "二寸 626x413", "方形头像 512x512"],
                value="一寸 413x626",
                label="输出尺寸",
            )
            id_button = gr.Button("生成证件照")
            id_gallery = gr.Gallery(label="证件照结果", columns=3, height=460)
            id_grid = gr.Image(type="filepath", label="证件照拼图")
            id_status = gr.Textbox(label="状态", lines=3)
            id_button.click(run_id_photo, inputs=[id_input, id_backgrounds, id_size], outputs=[id_gallery, id_grid, id_status])

        with gr.Tab("造型与姿态"):
            pose_input = gr.Image(type="filepath", label="上传人脸图片")
            pose_count = gr.Slider(minimum=4, maximum=16, step=4, value=8, label="生成数量")
            pose_strength = gr.Radio(choices=["标准", "更像本人"], value="标准", label="身份保持强度")
            pose_button = gr.Button("生成不同造型和姿态")
            pose_gallery = gr.Gallery(label="造型与姿态结果", columns=4, height=620)
            pose_grid = gr.Image(type="filepath", label="结果拼图")
            pose_ref_grid = gr.Image(type="filepath", label="姿态参考拼图")
            pose_status = gr.Textbox(label="状态", lines=3)
            pose_button.click(
                run_pose,
                inputs=[pose_input, pose_count, pose_strength],
                outputs=[pose_gallery, pose_grid, pose_ref_grid, pose_status],
            )

        with gr.Tab("项目成果展示"):
            showcase_button = gr.Button("加载已有成果")
            showcase_gallery = gr.Gallery(label="已有成果", columns=2, height=680)
            showcase_status = gr.Textbox(label="说明", lines=8)
            showcase_button.click(load_showcase, inputs=[], outputs=[showcase_gallery, showcase_status])

        gr.Markdown(
            """
            输出默认保存到 `outputs/facegan_studio/`，报告素材同步到 `report/report_assets/facegan_studio/`。
            证件照和身份保持生成均为课程技术演示结果，不作为正式证件或身份认证用途。
            """
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 FaceGAN Studio Web 应用")
    parser.add_argument("--project-root", default=None, help="项目根目录，默认自动使用当前目录")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", default=7860, type=int, help="监听端口")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_root(args.project_root, host=args.host, port=args.port, share=args.share)
    app = build_app(config)
    app.launch(server_name=config.host, server_port=config.port, share=config.share)


if __name__ == "__main__":
    main()
