"""Explicit RGB image/tensor conversion without implicit spatial transforms."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor


def image_to_tensor(image: Image.Image | Tensor) -> Tensor:
    """Return RGB BCHW float32 in [0,1], preserving size and batch.

    PIL inputs are EXIF-oriented and converted to RGB. Tensor inputs must already
    be floating RGB BCHW in [0,1]; integer tensors and ambiguous layouts fail.
    """
    if isinstance(image, Image.Image):
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255
    if not isinstance(image, Tensor):
        raise TypeError("image must be a PIL image or a floating BCHW tensor")
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError("image must have nonempty BCHW shape with 3 RGB channels")
    if not image.is_floating_point():
        raise TypeError("image tensor must be floating point in [0,1]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and in [0,1]")
    return image.to(dtype=torch.float32)


def tensor_to_images(images: Tensor) -> list[Image.Image]:
    """Convert validated RGB BCHW [0,1] tensors to individual uint8 PIL images."""
    batch = image_to_tensor(images).detach().cpu()
    pixels = (batch * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(array) for array in pixels]


def save_comparison(original: Tensor, reconstruction: Tensor, path: str | Path) -> None:
    """Save one original/reconstruction pair with English labels."""
    if original.shape != reconstruction.shape or original.shape[0] != 1:
        raise ValueError("comparison requires matching single-image BCHW tensors")
    left, right = tensor_to_images(original)[0], tensor_to_images(reconstruction)[0]
    from PIL import ImageDraw
    canvas = Image.new("RGB", (left.width * 2, left.height + 24), "white")
    canvas.paste(left, (0, 24))
    canvas.paste(right, (left.width, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "Original", fill="black")
    draw.text((left.width + 8, 6), "VAE reconstruction", fill="black")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
