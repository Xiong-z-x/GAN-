from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.dcgan.blocks import (
    MinibatchStdDev,
    ResidualDownsampleBlock,
    ResidualUpsampleBlock,
    SelfAttention2d,
    maybe_spectral_norm,
)


@dataclass(frozen=True)
class DCGANConfig:
    """DCGAN 与 DCGAN++ 网络结构参数。"""

    latent_dim: int = 100
    image_channels: int = 3
    generator_features: int = 64
    discriminator_features: int = 64
    image_size: int = 64
    architecture: str = "baseline"
    use_spectral_norm: bool = False
    use_attention: bool = False
    attention_resolutions: tuple[int, ...] = (32,)
    use_minibatch_stddev: bool = True
    max_channels: int = 512


class Generator(nn.Module):
    """原始 DCGAN 生成器，固定输出 64x64 图像。"""

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
    """原始 DCGAN 判别器，固定接收 64x64 图像。"""

    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        channels = config.image_channels
        features = config.discriminator_features
        use_sn = config.use_spectral_norm

        self.net = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(channels, features, 4, 2, 1, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Conv2d(features, features * 2, 4, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False), use_sn),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.net(images)
        return logits.view(images.size(0))


def _validate_image_size(image_size: int) -> None:
    if image_size < 64 or image_size & (image_size - 1):
        raise ValueError("image_size 必须是不小于 64 的 2 的幂。")


def _num_scale_blocks(image_size: int) -> int:
    _validate_image_size(image_size)
    return int(math.log2(image_size)) - 2


def _generator_schedule(image_size: int, base_channels: int, max_channels: int) -> tuple[int, list[int]]:
    blocks = _num_scale_blocks(image_size)
    start_channels = min(max_channels, base_channels * 8)
    min_channels = max(16, base_channels // 2)
    stage_channels: list[int] = []
    current = start_channels
    for _ in range(blocks):
        current = max(min_channels, current // 2)
        stage_channels.append(current)
    return start_channels, stage_channels


def _discriminator_schedule(image_size: int, base_channels: int, max_channels: int) -> tuple[int, list[int]]:
    blocks = _num_scale_blocks(image_size)
    start_channels = base_channels
    stage_channels: list[int] = []
    current = start_channels
    for _ in range(blocks):
        current = min(max_channels, current * 2)
        stage_channels.append(current)
    return start_channels, stage_channels


class ResidualGenerator(nn.Module):
    """DCGAN++ 生成器，使用残差上采样块。"""

    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.image_size = config.image_size
        start_channels, stage_channels = _generator_schedule(
            config.image_size,
            config.generator_features,
            config.max_channels,
        )

        self.project = nn.Linear(config.latent_dim, start_channels * 4 * 4)
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleDict()

        current_resolution = 4
        in_channels = start_channels
        attention_set = set(config.attention_resolutions)
        for out_channels in stage_channels:
            self.blocks.append(
                ResidualUpsampleBlock(
                    in_channels,
                    out_channels,
                    use_batch_norm=True,
                    use_spectral_norm=False,
                )
            )
            current_resolution *= 2
            if config.use_attention and current_resolution in attention_set:
                self.attentions[str(current_resolution)] = SelfAttention2d(out_channels)
            in_channels = out_channels

        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, config.image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        out = self.project(noise.view(noise.size(0), -1))
        out = out.view(noise.size(0), -1, 4, 4)
        current_resolution = 4
        for block in self.blocks:
            out = block(out)
            current_resolution *= 2
            key = str(current_resolution)
            if key in self.attentions:
                out = self.attentions[key](out)
        return self.to_rgb(out)


class ResidualDiscriminator(nn.Module):
    """DCGAN++ 判别器，使用残差下采样块。"""

    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        use_sn = config.use_spectral_norm
        start_channels, stage_channels = _discriminator_schedule(
            config.image_size,
            config.discriminator_features,
            config.max_channels,
        )

        self.from_rgb = maybe_spectral_norm(
            nn.Conv2d(config.image_channels, start_channels, kernel_size=3, padding=1),
            use_sn,
        )
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleDict()

        current_resolution = config.image_size
        in_channels = start_channels
        attention_set = set(config.attention_resolutions)
        for out_channels in stage_channels:
            self.blocks.append(
                ResidualDownsampleBlock(
                    in_channels,
                    out_channels,
                    use_spectral_norm=use_sn,
                )
            )
            current_resolution //= 2
            if config.use_attention and current_resolution in attention_set:
                self.attentions[str(current_resolution)] = SelfAttention2d(out_channels, use_spectral_norm=use_sn)
            in_channels = out_channels

        final_in_channels = in_channels + (1 if config.use_minibatch_stddev else 0)
        self.stddev = MinibatchStdDev() if config.use_minibatch_stddev else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.final_conv = maybe_spectral_norm(
            nn.Conv2d(final_in_channels, in_channels, kernel_size=3, padding=1),
            use_sn,
        )
        self.final_linear = maybe_spectral_norm(nn.Linear(in_channels * 4 * 4, 1), use_sn)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.from_rgb(images)
        current_resolution = self.image_size
        for block in self.blocks:
            out = block(out)
            current_resolution //= 2
            key = str(current_resolution)
            if key in self.attentions:
                out = self.attentions[key](out)
        out = self.stddev(out)
        out = self.activation(self.final_conv(out))
        out = out.reshape(out.size(0), -1)
        return self.final_linear(out).view(images.size(0))


def initialize_weights(module: nn.Module) -> None:
    """按 DCGAN 常用设置初始化卷积、线性和归一化层。"""

    class_name = module.__class__.__name__
    if "Conv" in class_name or "Linear" in class_name:
        weight = getattr(module, "weight_orig", None)
        if weight is None and hasattr(module, "weight") and module.weight is not None:
            weight = module.weight
        if weight is not None:
            nn.init.normal_(weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def build_models(config: DCGANConfig, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """构建并初始化生成器和判别器。"""

    if config.architecture == "baseline":
        if config.image_size != 64:
            raise ValueError("baseline 架构只支持 64x64。128 实验请使用 residual 架构。")
        generator: nn.Module = Generator(config)
        discriminator: nn.Module = Discriminator(config)
    elif config.architecture == "residual":
        generator = ResidualGenerator(config)
        discriminator = ResidualDiscriminator(config)
    else:
        raise ValueError(f"未知架构：{config.architecture}")

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    return generator, discriminator
