from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dcgan.augment import diff_augment
from src.dcgan.dataset import build_dataset
from src.dcgan.ema import ExponentialMovingAverage
from src.dcgan.losses import (
    bce_discriminator_loss,
    bce_generator_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    r1_penalty,
)
from src.dcgan.models import DCGANConfig, build_models
from src.dcgan.utils import append_jsonl, ensure_dir, save_json, save_samples, seed_everything


def parse_int_list(text: str) -> tuple[int, ...]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def parse_str_list(text: str) -> tuple[str, ...]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练手写 DCGAN / DCGAN++ 人像生成模型")
    parser.add_argument("--data-dir", type=Path, required=True, help="训练图像目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dcgan"), help="输出目录")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--image-size", type=int, default=64, help="图像分辨率")
    parser.add_argument("--latent-dim", type=int, default=100, help="随机噪声维度")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam 的 beta1")
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=20260430, help="随机种子")
    parser.add_argument("--sample-every", type=int, default=500, help="每隔多少步保存一次样例")
    parser.add_argument("--max-steps", type=int, default=0, help="调试用最大步数，0 表示不限制")
    parser.add_argument("--architecture", choices=("baseline", "residual"), default="baseline", help="模型架构")
    parser.add_argument("--loss", choices=("bce", "hinge"), default="bce", help="对抗损失类型")
    parser.add_argument("--generator-features", type=int, default=64, help="生成器基础通道数")
    parser.add_argument("--discriminator-features", type=int, default=64, help="判别器基础通道数")
    parser.add_argument("--max-channels", type=int, default=512, help="残差模型最大通道数")
    parser.add_argument("--use-spectral-norm", action="store_true", help="给判别器加谱归一化")
    parser.add_argument("--use-attention", action="store_true", help="在指定分辨率加入自注意力")
    parser.add_argument(
        "--attention-resolutions",
        type=parse_int_list,
        default=parse_int_list("32"),
        help="需要插入自注意力的分辨率，逗号分隔",
    )
    parser.add_argument("--dataset-augment", action="store_true", help="启用数据集级水平翻转")
    parser.add_argument("--diffaugment", type=parse_str_list, default=tuple(), help="判别器输入的可微增强策略")
    parser.add_argument("--use-ema", action="store_true", help="对生成器启用 EMA")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减系数")
    parser.add_argument("--use-r1", action="store_true", help="启用 R1 正则")
    parser.add_argument("--r1-gamma", type=float, default=10.0, help="R1 正则系数")
    parser.add_argument("--r1-interval", type=int, default=16, help="每隔多少步计算一次 R1")
    parser.add_argument("--generator-checkpoint", type=Path, default=None, help="生成器初始化权重")
    parser.add_argument("--discriminator-checkpoint", type=Path, default=None, help="判别器初始化权重")
    return parser.parse_args()


def load_checkpoint(module: torch.nn.Module, checkpoint_path: Path, device: torch.device, label: str) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} 权重不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    missing_keys, unexpected_keys = module.load_state_dict(checkpoint, strict=False)
    print(
        f"{label} 已加载：{checkpoint_path}，缺失键 {len(missing_keys)} 个，多余键 {len(unexpected_keys)} 个"
    )


def compute_discriminator_loss(loss_name: str, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    if loss_name == "bce":
        return bce_discriminator_loss(real_logits, fake_logits)
    if loss_name == "hinge":
        return hinge_discriminator_loss(real_logits, fake_logits)
    raise ValueError(f"未知损失类型：{loss_name}")


def compute_generator_loss(loss_name: str, fake_logits: torch.Tensor) -> torch.Tensor:
    if loss_name == "bce":
        return bce_generator_loss(fake_logits)
    if loss_name == "hinge":
        return hinge_generator_loss(fake_logits)
    raise ValueError(f"未知损失类型：{loss_name}")


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    samples_dir = ensure_dir(output_dir / "samples")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_path = output_dir / "loss_history.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_architecture = args.architecture
    if args.image_size > 64 and resolved_architecture == "baseline":
        print("当前分辨率高于 64，自动切换为 residual 架构。")
        resolved_architecture = "residual"

    dataset = build_dataset(args.data_dir, args.image_size, augment=args.dataset_augment)
    # 数据加载层只负责批量读取，尺寸统一和基础增强已经在 dataset.py 中完成。
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    config = DCGANConfig(
        latent_dim=args.latent_dim,
        image_channels=3,
        generator_features=args.generator_features,
        discriminator_features=args.discriminator_features,
        image_size=args.image_size,
        architecture=resolved_architecture,
        use_spectral_norm=args.use_spectral_norm,
        use_attention=args.use_attention,
        attention_resolutions=tuple(args.attention_resolutions),
        use_minibatch_stddev=True,
        max_channels=args.max_channels,
    )
    generator, discriminator = build_models(config, device)

    if args.generator_checkpoint is not None:
        load_checkpoint(generator, args.generator_checkpoint, device, "生成器")
    if args.discriminator_checkpoint is not None:
        load_checkpoint(discriminator, args.discriminator_checkpoint, device, "判别器")

    ema_generator = ExponentialMovingAverage(generator, decay=args.ema_decay).to(device) if args.use_ema else None
    if ema_generator is not None:
        ema_generator.update(generator)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    save_json(
        output_dir / "config.json",
        {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "lr": args.lr,
            "beta1": args.beta1,
            "seed": args.seed,
            "device": str(device),
            "dataset_size": len(dataset),
            "architecture": resolved_architecture,
            "loss": args.loss,
            "generator_features": args.generator_features,
            "discriminator_features": args.discriminator_features,
            "max_channels": args.max_channels,
            "use_spectral_norm": args.use_spectral_norm,
            "use_attention": args.use_attention,
            "attention_resolutions": list(args.attention_resolutions),
            "dataset_augment": args.dataset_augment,
            "diffaugment": list(args.diffaugment),
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "use_r1": args.use_r1,
            "r1_gamma": args.r1_gamma,
            "r1_interval": args.r1_interval,
        },
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # 交替更新判别器和生成器，保持 GAN 的基础训练节奏。
        for real_images in dataloader:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            optimizer_d.zero_grad(set_to_none=True)
            use_r1_now = args.use_r1 and global_step % max(1, args.r1_interval) == 0
            if use_r1_now:
                real_images_d = real_images.detach().requires_grad_(True)
            else:
                real_images_d = real_images.detach()

            if args.diffaugment:
                real_inputs = diff_augment(real_images_d, args.diffaugment)
            else:
                real_inputs = real_images_d

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_inputs = diff_augment(fake_images.detach(), args.diffaugment) if args.diffaugment else fake_images.detach()

            real_logits = discriminator(real_inputs)
            fake_logits = discriminator(fake_inputs)
            loss_d = compute_discriminator_loss(args.loss, real_logits, fake_logits)
            loss_r1 = torch.tensor(0.0, device=device)
            if use_r1_now:
                loss_r1 = r1_penalty(real_inputs, real_logits)
                loss_d = loss_d + 0.5 * args.r1_gamma * loss_r1
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_inputs = diff_augment(fake_images, args.diffaugment) if args.diffaugment else fake_images
            fake_logits_for_g = discriminator(fake_inputs)
            loss_g = compute_generator_loss(args.loss, fake_logits_for_g)
            loss_g.backward()
            optimizer_g.step()

            if ema_generator is not None:
                ema_generator.update(generator)

            global_step += 1
            log_item = {
                "epoch": epoch,
                "step": global_step,
                "loss_d": float(loss_d.item()),
                "loss_g": float(loss_g.item()),
                "loss_r1": float(loss_r1.item()) if use_r1_now else 0.0,
                "d_real": float(torch.sigmoid(real_logits).mean().item()),
                "d_fake": float(torch.sigmoid(fake_logits).mean().item()),
            }
            append_jsonl(logs_path, log_item)

            if global_step % args.sample_every == 0:
                sample_model = ema_generator.model if ema_generator is not None else generator
                # 固定噪声便于跨 epoch 对比生成质量变化。
                save_samples(sample_model, fixed_noise, samples_dir / f"step_{global_step:06d}.png")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        sample_model = ema_generator.model if ema_generator is not None else generator
        # 每轮结束都保存一次样例和 checkpoint，方便中途恢复和报告取图。
        save_samples(sample_model, fixed_noise, samples_dir / f"epoch_{epoch:03d}.png")
        torch.save(generator.state_dict(), checkpoints_dir / f"generator_epoch_{epoch:03d}.pth")
        torch.save(discriminator.state_dict(), checkpoints_dir / f"discriminator_epoch_{epoch:03d}.pth")
        if ema_generator is not None:
            torch.save(ema_generator.model.state_dict(), checkpoints_dir / f"generator_ema_epoch_{epoch:03d}.pth")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
