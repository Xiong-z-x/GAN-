from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def bce_discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """原始 DCGAN 的判别器损失。"""

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    return F.binary_cross_entropy_with_logits(real_logits, real_targets) + F.binary_cross_entropy_with_logits(
        fake_logits, fake_targets
    )


def bce_generator_loss(fake_logits: Tensor) -> Tensor:
    """原始 DCGAN 的生成器损失。"""

    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)


def hinge_discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Hinge 判别器损失。"""

    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_generator_loss(fake_logits: Tensor) -> Tensor:
    """Hinge 生成器损失。"""

    return -fake_logits.mean()


def r1_penalty(real_images: Tensor, real_logits: Tensor) -> Tensor:
    """R1 正则项。"""

    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return gradients.pow(2).sum(dim=1).mean()
