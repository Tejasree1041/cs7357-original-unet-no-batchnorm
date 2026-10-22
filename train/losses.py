"""Pixelwise cross entropy using Luke's precomputed border weight maps."""
import torch
import torch.nn.functional as F


def border_weighted_cross_entropy(logits, targets, weights=None):
    """logits [B,2,H,W], targets [B,H,W], weights [B,H,W].

    When weights=None, ordinary cross entropy supports the no-border ablation.
    The weights are already precomputed in data/processed/weights/*.npy.
    """
    if weights is None:
        return F.cross_entropy(logits, targets.long())
    per_pixel = F.cross_entropy(logits, targets.long(), reduction="none")
    if weights.shape != targets.shape:
        raise ValueError(f"Expected weight shape {targets.shape}, got {weights.shape}")
    return (per_pixel * weights.float()).mean()
