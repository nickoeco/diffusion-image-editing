"""Aligned binary/soft pixel masks and explicit latent-resolution conversion."""
from dataclasses import dataclass
import math
from numbers import Real
import warnings

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class PreparedMask:
    pixel: Tensor
    latent: Tensor


def validate_mask(mask: Tensor, batch: int, height: int, width: int) -> None:
    """Require exact floating B1HW shape, finite weights in [0,1]."""
    if not isinstance(mask, Tensor) or mask.shape != (batch, 1, height, width):
        raise ValueError("mask must have matching [B,1,H,W] shape")
    if not mask.is_floating_point() or not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("mask weights must be finite floating values in [0,1]")


def prepare_mask(mask: Image.Image | Tensor, image_size: tuple[int, int],
                  latent_size: tuple[int, int], batch_size: int = 1, *, binary: bool = True,
                  feather_radius: float = 0) -> PreparedMask:
    """Return B1HW pixel/latent masks, preserving alignment with oriented RGB.

    PIL masks must be L/1; tensors are HW or B1HW, floating [0,1] or boolean.
    A single mask broadcasts explicitly across an image batch. Pixel dimensions
    must already match; only latent conversion resizes. Binary uses threshold >=.5
    and nearest resizing, except feathered masks use bilinear like soft masks.
    Feather radius is Gaussian sigma in image pixels, truncated at ceil(3*sigma).
    """
    height, width = image_size
    if (type(batch_size) is not int or batch_size < 1 or min(*image_size, *latent_size) < 1
            or type(binary) is not bool):
        raise ValueError("mask sizes/batch must be positive; binary must be boolean")
    if (isinstance(feather_radius, bool) or not isinstance(feather_radius, Real)
            or not math.isfinite(feather_radius) or not 0 <= feather_radius <= max(image_size)):
        raise ValueError("feather radius must be finite, nonnegative and no larger than the image")
    if isinstance(mask, Image.Image):
        if mask.mode not in ("L", "1"):
            raise ValueError("PIL mask must be single-channel L or 1; convert explicitly")
        pixels = np.array(ImageOps.exif_transpose(mask).convert("L"), copy=True)
        mask = torch.from_numpy(pixels).float() / 255
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a grayscale PIL image or a tensor")
    if mask.dtype == torch.bool:
        mask = mask.float()
    elif not mask.is_floating_point():
        raise TypeError("tensor masks must be boolean or floating point in [0,1]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim != 4 or mask.shape[0] not in (1, batch_size):
        raise ValueError("mask must be HW or B1HW, with batch 1 or image batch size")
    validate_mask(mask, mask.shape[0], height, width)
    pixel = mask.detach().float().clone()
    if pixel.shape[0] == 1 and batch_size != 1:
        pixel = pixel.expand(batch_size, -1, -1, -1).clone()
    if binary:
        pixel = (pixel >= 0.5).float()
    constant = pixel.amin(dim=(1, 2, 3), keepdim=True) == pixel.amax(dim=(1, 2, 3), keepdim=True)
    if feather_radius > 0 and not torch.all(constant):
        unblurred = pixel
        radius = math.ceil(3 * feather_radius)
        positions = torch.arange(-radius, radius + 1, device=pixel.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (positions / feather_radius).square())
        kernel = kernel / kernel.sum()
        # Replicate padding preserves constant zero/full masks at image boundaries.
        pixel = F.conv2d(F.pad(pixel, (radius, radius, 0, 0), mode="replicate"), kernel.reshape(1, 1, 1, -1))
        pixel = F.conv2d(F.pad(pixel, (0, 0, radius, radius), mode="replicate"), kernel.reshape(1, 1, -1, 1)).clamp(0, 1)
        pixel = torch.where(constant, unblurred, pixel)
    if binary and feather_radius == 0:
        latent = F.interpolate(pixel, size=latent_size, mode="nearest")
    else:
        latent = F.interpolate(pixel, size=latent_size, mode="bilinear", align_corners=False)
    if torch.any(pixel > 0) and not torch.any(latent > 0):
        warnings.warn("Mask disappears at latent resolution; enlarge it or use a soft mask", UserWarning, stacklevel=2)
    return PreparedMask(pixel, latent)
