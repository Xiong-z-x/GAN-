from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _rand_factor(batch: int, device: torch.device, dtype: torch.dtype, low: float, high: float) -> Tensor:
    return torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * (high - low) + low


def color_augment(images: Tensor) -> Tensor:
    """颜色扰动，包含亮度、对比度和饱和度。"""

    batch = images.size(0)
    device = images.device
    dtype = images.dtype

    brightness = _rand_factor(batch, device, dtype, -0.2, 0.2)
    images = images + brightness

    mean = images.mean(dim=1, keepdim=True)
    saturation = _rand_factor(batch, device, dtype, 0.0, 2.0)
    images = (images - mean) * saturation + mean

    global_mean = images.mean(dim=(1, 2, 3), keepdim=True)
    contrast = _rand_factor(batch, device, dtype, 0.75, 1.25)
    images = (images - global_mean) * contrast + global_mean
    return images


def translation_augment(images: Tensor, ratio: float = 0.125) -> Tensor:
    """平移增强。"""

    batch, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    shift_y = torch.randint(
        low=-max(1, int(round(height * ratio))),
        high=max(1, int(round(height * ratio))) + 1,
        size=(batch,),
        device=device,
    )
    shift_x = torch.randint(
        low=-max(1, int(round(width * ratio))),
        high=max(1, int(round(width * ratio))) + 1,
        size=(batch,),
        device=device,
    )

    theta = torch.zeros((batch, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = shift_x.to(dtype) * 2.0 / max(1, width)
    theta[:, 1, 2] = shift_y.to(dtype) * 2.0 / max(1, height)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)


def cutout_augment(images: Tensor, ratio: float = 0.25) -> Tensor:
    """随机擦除增强。"""

    batch, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    cutout_h = max(1, int(round(height * ratio)))
    cutout_w = max(1, int(round(width * ratio)))
    center_y = torch.randint(0, height, (batch, 1, 1, 1), device=device)
    center_x = torch.randint(0, width, (batch, 1, 1, 1), device=device)

    yy = torch.arange(height, device=device).view(1, 1, height, 1)
    xx = torch.arange(width, device=device).view(1, 1, 1, width)
    mask = (yy < center_y - cutout_h // 2) | (yy > center_y + cutout_h // 2) | (xx < center_x - cutout_w // 2) | (xx > center_x + cutout_w // 2)
    return images * mask.to(dtype)


def diff_augment(images: Tensor, policy: Sequence[str] = ("color", "translation", "cutout")) -> Tensor:
    """对抗训练时使用的轻量可微增强。"""

    augmented = images
    for item in policy:
        if item == "color":
            augmented = color_augment(augmented)
        elif item == "translation":
            augmented = translation_augment(augmented)
        elif item == "cutout":
            augmented = cutout_augment(augmented)
    return augmented
