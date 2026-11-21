"""Explicit outpainting canvas placement, initialization and editable overlap."""
from dataclasses import dataclass

from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F

from diffusion_editor.utils.image_utils import image_to_tensor


@dataclass(frozen=True)
class OutpaintCanvas:
    pixels: Tensor
    mask: Tensor
    source_box: tuple[int, int, int, int]
    margins: tuple[int, int, int, int]
    overlap: int
    fill: str


def prepare_outpaint_canvas(image: Image.Image | Tensor, *, left: int = 0, right: int = 0,
                            top: int = 0, bottom: int = 0, overlap: int = 0,
                            fill: str = "edge") -> OutpaintCanvas:
    """BCHW [0,1] canvas; mask 1 generates, 0 preserves. No resize/crop.

    Overlap is a linear editable band INSIDE original edges that have expansion.
    Default zero preserves the whole source rectangle under final compositing.
    """
    margins = (left, right, top, bottom)
    if any(type(v) is not int or v < 0 for v in (*margins, overlap)) or not any(margins):
        raise ValueError("Margins/overlap must be nonnegative integers; expand at least one side")
    if fill not in ("edge", "neutral"):
        raise ValueError("Canvas fill must be edge or neutral")
    pixels = image_to_tensor(image).detach().clone()
    batch, _, height, width = pixels.shape
    if 2 * overlap >= min(height, width):
        raise ValueError("Overlap must be less than half the smaller source dimension")
    canvas = F.pad(pixels, margins, mode="replicate") if fill == "edge" else F.pad(pixels, margins, value=0.5)
    mask = torch.ones((batch, 1, height + top + bottom, width + left + right), device=pixels.device)
    interior = torch.zeros((height, width), device=pixels.device)
    if overlap:
        x = torch.arange(width, device=pixels.device).float()[None, :].expand(height, -1)
        y = torch.arange(height, device=pixels.device).float()[:, None].expand(-1, width)
        distances = ((left, x), (right, width - 1 - x), (top, y), (bottom, height - 1 - y))
        for margin, distance in distances:
            if margin:
                interior = torch.maximum(interior, (1 - distance / overlap).clamp(0, 1))
    mask[:, :, top:top + height, left:left + width] = interior
    return OutpaintCanvas(canvas, mask, (left, top, left + width, top + height), margins, overlap, fill)
