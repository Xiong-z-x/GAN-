from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    """固定随机种子，便于复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回路径对象。"""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """保存缩进后的 JSON 文件。"""

    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    """追加一行 JSONL 训练日志。"""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


@torch.no_grad()
def save_samples(
    generator: torch.nn.Module,
    fixed_noise: torch.Tensor,
    output_path: str | Path,
    *,
    nrow: int = 8,
) -> None:
    """保存固定噪声生成的样例图。"""

    was_training = generator.training
    generator.eval()
    fake_images = generator(fixed_noise).detach().cpu()
    ensure_dir(Path(output_path).parent)
    save_image(fake_images, output_path, normalize=True, value_range=(-1, 1), nrow=nrow)
    generator.train(was_training)
