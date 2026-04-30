from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset[Tensor]):
    """递归读取目录中的图像文件，不要求按类别分文件夹。"""

    def __init__(self, root: str | Path, transform: Callable[[Image.Image], Tensor]) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise FileNotFoundError(f"未在目录中找到图像文件：{self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def build_transform(image_size: int) -> transforms.Compose:
    """构建 DCGAN 训练使用的图像变换。"""

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_dataset(root: str | Path, image_size: int) -> FlatImageDataset:
    """创建人脸图像数据集。"""

    return FlatImageDataset(root=root, transform=build_transform(image_size))

