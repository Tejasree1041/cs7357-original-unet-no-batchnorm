"""Minimal training skeleton; evaluation, checkpoints and logging belong to teammates.

Run from repository root:
  python -m train.trainer --model full --epochs 1 --batch-size 1
  python -m train.trainer --model no_skips --epochs 1 --batch-size 1
  python -m train.trainer --model no_elastic --epochs 1 --batch-size 1
  python -m train.trainer --model no_border --epochs 1 --batch-size 1
"""
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from train.dataset import NucleiDataset
from train.augment import NucleiAugment
from train.losses import border_weighted_cross_entropy


def train_one_epoch(model, loader, optimizer, device, use_border_weight=True):
    model.train()
    total = 0.0
    for images, masks, weights in loader:
        images, masks = images.to(device), masks.to(device)
        weights = weights.to(device) if use_border_weight else None
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = border_weighted_cross_entropy(logits, masks, weights)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["full", "no_skips", "no_elastic", "no_border"], default="full")
    parser.add_argument("--split-file", default="data/splits/split_seed42.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    ds = NucleiDataset(args.split_file, split="train", processed_dir=args.processed_dir,
                       transform=NucleiAugment(elastic=args.model != "no_elastic"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = UNet(use_skip=args.model != "no_skips").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"model={args.model}, device={device}, train_images={len(ds)}")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device,
                               use_border_weight=args.model != "no_border")
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={loss:.5f}")
    # Teammates will connect validation, checkpointing, metrics, and result logging.


if __name__ == "__main__":
    main()
