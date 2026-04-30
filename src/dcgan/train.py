from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dcgan.dataset import build_dataset
from src.dcgan.models import DCGANConfig, build_models
from src.dcgan.utils import append_jsonl, ensure_dir, save_json, save_samples, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练手写 DCGAN 人像生成 baseline")
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
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    samples_dir = ensure_dir(output_dir / "samples")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_path = output_dir / "loss_history.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args.data_dir, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    config = DCGANConfig(latent_dim=args.latent_dim)
    generator, discriminator = build_models(config, device)
    criterion = nn.BCEWithLogitsLoss()
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
        },
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for real_images in dataloader:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_labels)

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_logits, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            generator.zero_grad(set_to_none=True)
            logits_for_g = discriminator(fake_images)
            loss_g = criterion(logits_for_g, real_labels)
            loss_g.backward()
            optimizer_g.step()

            global_step += 1
            log_item = {
                "epoch": epoch,
                "step": global_step,
                "loss_d": float(loss_d.item()),
                "loss_g": float(loss_g.item()),
                "d_real": float(torch.sigmoid(real_logits).mean().item()),
                "d_fake": float(torch.sigmoid(fake_logits).mean().item()),
            }
            append_jsonl(logs_path, log_item)

            if global_step % args.sample_every == 0:
                save_samples(generator, fixed_noise, samples_dir / f"step_{global_step:06d}.png")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        save_samples(generator, fixed_noise, samples_dir / f"epoch_{epoch:03d}.png")
        torch.save(generator.state_dict(), checkpoints_dir / f"generator_epoch_{epoch:03d}.pth")
        torch.save(discriminator.state_dict(), checkpoints_dir / f"discriminator_epoch_{epoch:03d}.pth")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

