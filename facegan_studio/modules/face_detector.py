from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .image_utils import load_rgb, save_rgb


@dataclass(frozen=True)
class FaceDetectionResult:
    image_path: Path
    preview_path: Path
    bbox: tuple[int, int, int, int]
    source: str
    message: str


def _center_bbox(width: int, height: int) -> tuple[int, int, int, int]:
    side = int(min(width, height) * 0.62)
    left = (width - side) // 2
    top = max(int(height * 0.18), 0)
    return left, top, left + side, top + side


def _detect_with_opencv(image_path: Path) -> tuple[int, int, int, int] | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(x + w), int(y + h)


def detect_face_preview(image_path: str | Path, output_dir: str | Path) -> FaceDetectionResult:
    input_path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(input_path)
    bbox = _detect_with_opencv(input_path)
    source = "opencv"
    message = "已使用 OpenCV 检测到人脸。"

    if bbox is None:
        bbox = _center_bbox(*image.size)
        source = "fallback"
        message = "未检测到明确人脸，已使用中心区域作为保守预览。"

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(bbox, outline=(0, 220, 80), width=max(3, image.width // 160))
    preview_path = save_rgb(preview, out_dir / "face_detected_preview.png")

    return FaceDetectionResult(
        image_path=input_path,
        preview_path=preview_path,
        bbox=bbox,
        source=source,
        message=message,
    )
