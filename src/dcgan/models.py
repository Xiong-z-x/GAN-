from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DCGANConfig:
    """DCGAN 网络结构参数。"""

    latent_dim: int = 100
    image_channels: int = 3
    generator_features: int = 64
    discriminator_features: int = 64


class Generator(nn.Module):
    """将随机噪声映射为 64x64 人脸图像。"""

    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        z_dim = config.latent_dim
        channels = config.image_channels
        features = config.generator_features

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    """判断输入图像是真实样本还是生成样本。"""

    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        channels = config.image_channels
        features = config.discriminator_features

        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.net(images)
        return logits.view(images.size(0))


def initialize_weights(module: nn.Module) -> None:
    """按 DCGAN 论文常用设置初始化卷积和归一化层。"""

    class_name = module.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def build_models(config: DCGANConfig, device: torch.device) -> tuple[Generator, Discriminator]:
    """构建并初始化生成器和判别器。"""

    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    return generator, discriminator

