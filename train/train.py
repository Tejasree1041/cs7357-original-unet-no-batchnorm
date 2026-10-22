import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from train.dataset import NucleiDataset
import os

def dice_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_ds = NucleiDataset("data/splits/split_seed42.json", "train")
    val_ds = NucleiDataset("data/splits/split_seed42.json", "val")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    # Model
    model = UNet(use_skip=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_dice = 0.0
    os.makedirs("results", exist_ok=True)

    for epoch in range(50):
        model.train()
        for imgs, masks, weights in train_loader:
            imgs, masks, weights = imgs.to(device), masks.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_dice += dice_score(logits, masks).item()
        val_dice /= len(val_loader)
        print(f"Epoch {epoch+1}/50 - Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "results/best_unet.pth")
            print("  Saved new best model")

if __name__ == "__main__":
    main()
