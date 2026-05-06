from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    """按需给层加谱归一化。"""

    return spectral_norm(module) if enabled else module


class SelfAttention2d(nn.Module):
    """SAGAN 风格的二维自注意力层。"""

    def __init__(self, channels: int, *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        inner_channels = max(1, channels // 8)
        self.query = maybe_spectral_norm(nn.Conv2d(channels, inner_channels, kernel_size=1), use_spectral_norm)
        self.key = maybe_spectral_norm(nn.Conv2d(channels, inner_channels, kernel_size=1), use_spectral_norm)
        self.value = maybe_spectral_norm(nn.Conv2d(channels, channels, kernel_size=1), use_spectral_norm)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        query = self.query(x).reshape(batch_size, -1, height * width).transpose(1, 2)
        key = self.key(x).reshape(batch_size, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).reshape(batch_size, channels, height * width)
        attended = torch.bmm(value, attention.transpose(1, 2)).reshape(batch_size, channels, height, width)
        return x + self.gamma * attended


class ResidualUpsampleBlock(nn.Module):
    """带上采样的残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_batch_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = maybe_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), use_spectral_norm)
        self.conv2 = maybe_spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), use_spectral_norm)
        self.skip = maybe_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1), use_spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(self.upsample(x))
        out = self.norm1(x)
        out = self.activation(out)
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + residual


class ResidualDownsampleBlock(nn.Module):
    """带下采样的残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.conv1 = maybe_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), use_spectral_norm)
        self.conv2 = maybe_spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), use_spectral_norm)
        self.skip = maybe_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1), use_spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.avg_pool2d(self.skip(x), kernel_size=2)
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = F.avg_pool2d(out, kernel_size=2)
        return out + residual


class MinibatchStdDev(nn.Module):
    """为判别器最后阶段补充 batch 统计量。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 1:
            std_map = torch.zeros((1, 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
            return torch.cat([x, std_map], dim=1)

        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8)
        mean_std = std.mean().view(1, 1, 1, 1)
        mean_std = mean_std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)
