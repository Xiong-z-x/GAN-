from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    """生成器参数的指数滑动平均。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = deepcopy(model)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def to(self, device: torch.device | str) -> "ExponentialMovingAverage":
        self.model.to(device)
        return self

    @torch.no_grad()
    def update(self, source_model: nn.Module) -> None:
        source_state = source_model.state_dict()
        ema_state = self.model.state_dict()
        for key, ema_value in ema_state.items():
            source_value = source_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)
