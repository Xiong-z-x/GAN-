from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .image_utils import collect_images, copy_files, load_rgb, make_grid, save_rgb, write_metadata
from .paths import ProjectPaths, current_timestamp


POSE_PROMPTS = [
    ("black_glasses", "a realistic portrait photo of the same person wearing black frame glasses, natural light, detailed face"),
    ("metal_glasses", "a realistic portrait photo of the same person wearing thin metal glasses, clean background, detailed face"),
    ("business", "a realistic portrait photo of the same person wearing a dark suit, professional headshot, clean background"),
    ("casual", "a realistic portrait photo of the same person wearing casual clothes, relaxed expression, soft light"),
    ("side_view", "a realistic portrait photo of the same person looking sideways, half body portrait, natural skin texture"),
    ("smile", "a realistic portrait photo of the same person smiling gently, bright natural light, detailed face"),
    ("cinematic", "a cinematic portrait photo of the same person, confident pose, realistic skin, shallow depth of field"),
    ("outdoor", "a realistic portrait photo of the same person walking outdoors, dynamic pose, street photography"),
    ("studio", "a realistic studio portrait photo of the same person, sharp focus, simple background"),
    ("coat", "a realistic portrait photo of the same person wearing a coat, elegant style, natural face"),
    ("profile", "a realistic portrait photo of the same person in a three quarter view, soft light"),
    ("closeup", "a close-up realistic portrait photo of the same person, detailed eyes, natural skin"),
    ("workplace", "a realistic portrait photo of the same person in an office, professional style"),
    ("travel", "a realistic portrait photo of the same person during travel, natural expression"),
    ("soft_light", "a realistic portrait photo of the same person, soft light, clean composition"),
    ("half_body", "a realistic half body portrait photo of the same person, balanced pose"),
]


NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, deformed, distorted face, bad anatomy, "
    "bad hands, extra fingers, missing fingers, duplicate face, watermark, text, logo"
)


@dataclass(frozen=True)
class PoseStyleResult:
    output_dir: Path
    image_paths: list[Path]
    grid_path: Path
    reference_grid_path: Path
    metadata_path: Path


def find_base_model_dir() -> Path:
    candidates = [
        os.environ.get("INSTANTID_BASE_MODEL_DIR"),
        "/root/autodl-fs/models/YamerMIX_v8",
        "/autodl-fs/data/models/YamerMIX_v8",
        "/root/autodl-fs/data/models/YamerMIX_v8",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "model_index.json").exists():
            return path
    raise FileNotFoundError("未找到 SDXL 基础模型目录，请设置 INSTANTID_BASE_MODEL_DIR。")


def _collect_pose_references(paths: ProjectPaths, limit: int) -> list[Path]:
    refs: list[Path] = []
    refs.extend(collect_images(paths.project_root / "data" / "raw" / "my_photos", recursive=True, limit=8))
    refs.extend(collect_images(paths.showcase_results_dir / "outputs" / "gan_showcase" / "stylegan3_ffhq_best", recursive=False, limit=32))
    refs.extend(collect_images(paths.project_root / "outputs" / "stylegan3" / "images", recursive=False, limit=32))
    refs.extend(collect_images(paths.project_root / "data" / "raw" / "celeba", recursive=True, limit=64))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in refs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique[:limit]


def generate_pose_styles(
    image_path: str | Path,
    project_root: str | Path | None = None,
    count: int = 8,
    identity_strength: str = "standard",
    timestamp: str | None = None,
) -> PoseStyleResult:
    paths = ProjectPaths(project_root)
    stamp = timestamp or current_timestamp()
    run_dir = paths.create_run_dir("pose_style", stamp)
    report_dir = paths.create_report_run_dir("pose_style", stamp)
    input_path = Path(image_path)

    instantid_root = paths.instantid_repo
    if not (instantid_root / "pipeline_stable_diffusion_xl_instantid.py").exists():
        raise FileNotFoundError(f"未找到 InstantID 仓库：{instantid_root}")
    if not (instantid_root / "checkpoints" / "ControlNetModel" / "diffusion_pytorch_model.safetensors").exists():
        raise FileNotFoundError("未找到 InstantID ControlNet 权重。")
    if not (instantid_root / "checkpoints" / "ip-adapter.bin").exists():
        raise FileNotFoundError("未找到 InstantID ip-adapter.bin。")

    sys.path.insert(0, str(instantid_root))

    import cv2
    import numpy as np
    import torch
    from diffusers import ControlNetModel
    from diffusers.utils import load_image
    from insightface.app import FaceAnalysis
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = find_base_model_dir()

    app = FaceAnalysis(
        name="antelopev2",
        root=str(instantid_root),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    face_image = load_image(str(input_path)).convert("RGB")
    face_bgr = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    faces = app.get(face_bgr)
    if not faces:
        raise RuntimeError("输入图片没有检测到可用人脸。")

    face_info = max(
        faces,
        key=lambda face: float((face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])),
    )
    face_emb = face_info["embedding"]

    pose_refs = _collect_pose_references(paths, max(count, 4))
    pose_items = []
    for ref_path in pose_refs:
        ref_image = load_image(str(ref_path)).convert("RGB")
        ref_bgr = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        ref_faces = app.get(ref_bgr)
        if not ref_faces:
            continue
        ref_face = max(
            ref_faces,
            key=lambda face: float((face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])),
        )
        pose_items.append((ref_path, ref_image, ref_face))
        if len(pose_items) >= count:
            break

    if not pose_items:
        pose_items.append((input_path, face_image, face_info))

    controlnet = ControlNetModel.from_pretrained(
        str(instantid_root / "checkpoints" / "ControlNetModel"),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        str(base_model),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(str(instantid_root / "checkpoints" / "ip-adapter.bin"))
    pipe.enable_vae_tiling()

    if identity_strength == "strong":
        ip_adapter_scale = 0.9
        control_scale = 0.75
    else:
        ip_adapter_scale = 0.82
        control_scale = 0.8

    generator = torch.Generator(device=device).manual_seed(20260506)
    result_paths: list[Path] = []
    reference_paths: list[Path] = []

    save_rgb(face_image, run_dir / "identity_reference.png")

    for index in range(count):
        prompt_key, prompt = POSE_PROMPTS[index % len(POSE_PROMPTS)]
        ref_path, ref_image, ref_face = pose_items[index % len(pose_items)]
        kps_image = draw_kps(ref_image, ref_face["kps"])
        ref_save_path = save_rgb(ref_image, run_dir / f"pose_reference_{index:02d}.png")
        reference_paths.append(ref_save_path)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image_embeds=face_emb,
            image=kps_image,
            controlnet_conditioning_scale=control_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=28,
            guidance_scale=5.0,
            generator=generator,
            width=1024,
            height=1024,
        ).images[0]

        out_path = run_dir / f"pose_style_{index:02d}_{prompt_key}.png"
        image.save(out_path)
        result_paths.append(out_path)

    result_grid = make_grid([load_rgb(path) for path in result_paths], columns=4, cell_size=(256, 256), padding=0)
    grid_path = save_rgb(result_grid, run_dir / "pose_style_grid.png")
    reference_grid = make_grid([load_rgb(path) for path in reference_paths], columns=4, cell_size=(256, 256), padding=0)
    reference_grid_path = save_rgb(reference_grid, run_dir / "pose_reference_grid.png")

    copied = copy_files([run_dir / "identity_reference.png", *reference_paths, *result_paths, grid_path, reference_grid_path], report_dir)
    metadata_path = write_metadata(
        run_dir / "metadata.json",
        {
            "mode": "pose_style",
            "input": str(input_path),
            "base_model": str(base_model),
            "count": count,
            "identity_strength": identity_strength,
            "outputs": [str(path) for path in result_paths],
            "pose_references": [str(path) for path in reference_paths],
            "report_assets": [str(path) for path in copied],
            "note": "InstantID 固定输入人脸身份向量，生成不同姿态和造型；结果仍需人工筛选身份一致性。",
        },
    )
    shutil.copy2(metadata_path, report_dir / "metadata.json")

    return PoseStyleResult(
        output_dir=run_dir,
        image_paths=result_paths,
        grid_path=grid_path,
        reference_grid_path=reference_grid_path,
        metadata_path=metadata_path,
    )
