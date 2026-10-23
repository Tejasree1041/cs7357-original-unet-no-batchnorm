"""Standalone diagnostic for original U-Net without BatchNorm."""

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from train.dataset import NucleiDataset
from train.augment import NucleiAugment
from train.losses import border_weighted_cross_entropy


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_intersection = 0
    total_predicted = 0
    total_target = 0
    total_pixels = 0

    batch_dice_sum = 0.0
    batch_count = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        intersection = ((predictions == 1) & (masks == 1)).sum().item()
        predicted = (predictions == 1).sum().item()
        target = (masks == 1).sum().item()

        # Original global Dice
        total_intersection += intersection
        total_predicted += predicted
        total_target += target
        total_pixels += masks.numel()

        # Luke's batch-averaged Dice
        batch_dice = (2.0 * intersection) / (predicted + target + 1e-8)
        batch_dice_sum += batch_dice
        batch_count += 1

    global_dice = (2.0 * total_intersection) / (total_predicted + total_target + 1e-8)

    mean_batch_dice = batch_dice_sum / max(batch_count, 1)

    foreground_pct = 100 * total_predicted / max(total_pixels, 1)
    target_pct = 100 * total_target / max(total_pixels, 1)

    return global_dice, mean_batch_dice, foreground_pct, target_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_ds = NucleiDataset(
        "data/splits/split_seed42.json",
        split="train",
        transform=NucleiAugment(elastic=True),
    )

    val_ds = NucleiDataset(
        "data/splits/split_seed42.json",
        split="val",
        transform=None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = UNet(use_skip=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_dir = Path("checkpoints/no_batchnorm")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    best_epoch = 0

    print(
        f"device={device}, train={len(train_ds)}, "
        f"validation={len(val_ds)}, batch_size={args.batch_size}"
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, masks, weights in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            weights = weights.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = border_weighted_cross_entropy(logits, masks, weights)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        global_dice, batch_dice, pred_fg, true_fg = validate(model, val_loader, device)
        if batch_dice > best_dice:
            best_dice = batch_dice
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_batch_dice": best_dice,
                    "best_val_global_dice": global_dice,
                    "seed": args.seed,
                },
                checkpoint_dir / "best_model.pt",
            )

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_dice={global_dice:.4f} "
            f"batch_dice={batch_dice:.4f} "
            f"pred_fg={pred_fg:.2f}% "
            f"true_fg={true_fg:.2f}%",
            flush=True,
        )
    print(
        f"Best checkpoint: epoch={best_epoch}, val_batch_dice={best_dice:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
