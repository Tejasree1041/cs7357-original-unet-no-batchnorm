"""Evaluate the best original U-Net checkpoint on held-out test data."""

import argparse
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from train.dataset import NucleiDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_intersection = 0
    total_predicted = 0
    total_target = 0

    batch_dice_scores = []
    image_dice_scores = []

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        pred_fg = predictions == 1
        true_fg = masks == 1

        intersection = (pred_fg & true_fg).sum().item()
        predicted = pred_fg.sum().item()
        target = true_fg.sum().item()

        total_intersection += intersection
        total_predicted += predicted
        total_target += target

        # Same batch-averaged Dice calculation as Luke.
        batch_dice = (
            2.0 * intersection / (predicted + target + 1e-8)
        )
        batch_dice_scores.append(batch_dice)

        # Additional per-image Dice metric.
        for p, t in zip(pred_fg, true_fg):
            i = (p & t).sum().item()
            p_count = p.sum().item()
            t_count = t.sum().item()

            image_dice = (
                2.0 * i / (p_count + t_count + 1e-8)
            )
            image_dice_scores.append(image_dice)

    global_dice = (
        2.0 * total_intersection /
        (total_predicted + total_target + 1e-8)
    )

    mean_batch_dice = (
        sum(batch_dice_scores) / len(batch_dice_scores)
    )

    mean_image_dice = (
        sum(image_dice_scores) / len(image_dice_scores)
    )

    return global_dice, mean_batch_dice, mean_image_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/no_batchnorm/best_model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = UNet(use_skip=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_ds = NucleiDataset(
        "data/splits/split_seed42.json",
        split="test",
        transform=None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    global_dice, batch_dice, image_dice = evaluate(
        model, test_loader, device
    )

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Best validation batch Dice: "
        f"{checkpoint['best_val_batch_dice']:.4f}"
    )
    print(f"Test images: {len(test_ds)}")
    print(f"Test global Dice: {global_dice:.4f}")
    print(f"Test batch-averaged Dice: {batch_dice:.4f}")
    print(f"Test mean per-image Dice: {image_dice:.4f}")


if __name__ == "__main__":
    main()
