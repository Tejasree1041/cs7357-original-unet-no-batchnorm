"""Synchronized image/mask/weight spatial augmentation for training only."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class NucleiAugment:
    def __init__(self, elastic=True, alpha=20.0, sigma=4.0):
        self.elastic = elastic
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, mask, weight):
        # Input: 2-D numpy arrays; output same shapes and dtypes.
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        if image.shape != mask.shape or mask.shape != weight.shape:
            raise ValueError("image, mask and weight must have identical H x W shapes")

        if np.random.rand() < 0.5:
            image, mask, weight = (np.fliplr(a) for a in (image, mask, weight))
        if np.random.rand() < 0.5:
            image, mask, weight = (np.flipud(a) for a in (image, mask, weight))
        k = np.random.randint(0, 4)
        image, mask, weight = (np.rot90(a, k) for a in (image, mask, weight))

        if self.elastic:
            h, w = image.shape
            # Same displacement field for all three arrays; nearest for discrete mask.
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="reflect") * self.alpha
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="reflect") * self.alpha
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            coords = [yy + dy, xx + dx]
            image = map_coordinates(image, coords, order=1, mode="reflect")
            mask = map_coordinates(mask, coords, order=0, mode="nearest")
            weight = map_coordinates(weight, coords, order=1, mode="reflect")

        # Grayscale shifts affect the image only.
        image = np.clip(image * np.random.uniform(0.9, 1.1)
                        + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        return (np.ascontiguousarray(image, dtype=np.float32),
                np.ascontiguousarray(mask, dtype=np.int64),
                np.ascontiguousarray(weight, dtype=np.float32))
