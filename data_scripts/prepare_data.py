import os
import json
import random
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from pathlib import Path
from tqdm import tqdm
from skimage.segmentation import find_boundaries

# ---------- Config ----------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")
TARGET_SIZE = (256, 256)
SEED = 42
W0 = 10
SIGMA = 5

OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)
(OUT_DIR / "masks").mkdir(exist_ok=True)
(OUT_DIR / "weights").mkdir(exist_ok=True)

def merge_masks(mask_dir):
    """Merge all instance masks into one binary mask."""
    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        return None
    first = np.array(Image.open(masks[0]))
    binary = np.zeros_like(first, dtype=np.uint8)
    for m in masks:
        binary = np.maximum(binary, (np.array(Image.open(m)) > 0).astype(np.uint8))
    return binary

def compute_weight_map(binary_mask, w0=10, sigma=5):
    """U-Net paper border weight map (practical version)."""
    # Class balance weight
    fg = binary_mask.sum()
    total = binary_mask.size
    wc = np.ones_like(binary_mask, dtype=np.float32)
    if total > 0:
        wc[binary_mask == 1] = total / (2 * fg) if fg > 0 else 1.0
        wc[binary_mask == 0] = total / (2 * (total - fg)) if (total - fg) > 0 else 1.0

    # Border weight
    boundaries = find_boundaries(binary_mask, mode="outer")
    if boundaries.any():
        dist = distance_transform_edt(~boundaries)
        border_term = w0 * np.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        border_term = np.zeros_like(binary_mask, dtype=np.float32)

    weight = wc + border_term
    return weight.astype(np.float32)

def process_one(image_id):
    img_folder = RAW_DIR / image_id / "images"
    mask_dir = RAW_DIR / image_id / "masks"
    img_files = list(img_folder.glob("*.png"))
    if not img_files:
        return False

    # Load and resize image (grayscale)
    img = Image.open(img_files[0]).convert("L")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_arr = np.array(img)

    # Merge and resize masks
    binary = merge_masks(mask_dir)
    if binary is None:
        return False
    binary_img = Image.fromarray((binary * 255).astype(np.uint8))
    binary_img = binary_img.resize(TARGET_SIZE, Image.NEAREST)
    binary = (np.array(binary_img) > 127).astype(np.uint8)

    # Weight map
    weight = compute_weight_map(binary, W0, SIGMA)
    np.save(OUT_DIR / "weights" / f"{image_id}.npy", weight)

    # Save image and mask
    Image.fromarray(img_arr).save(OUT_DIR / "images" / f"{image_id}.png")
    Image.fromarray((binary * 255).astype(np.uint8)).save(OUT_DIR / "masks" / f"{image_id}.png")
    return True

def main():
    image_ids = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(image_ids)} images")

    success = []
    for iid in tqdm(image_ids):
        if process_one(iid):
            success.append(iid)

    print(f"Successfully processed {len(success)} images")

    # Fixed split
    random.seed(SEED)
    random.shuffle(success)
    train_ids = success[:30]
    val_ids = success[30:30+67]
    test_ids = success[30+67:30+67+67]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(SPLIT_DIR / "split_seed42.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("Split saved:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val:   {len(val_ids)}")
    print(f"  Test:  {len(test_ids)}")

if __name__ == "__main__":
    main()
