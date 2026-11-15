"""Labelled grids for reconstruction and explicitly noisy latent previews."""
from collections.abc import Sequence
import math
from pathlib import Path

from PIL import Image, ImageDraw
from torch import Tensor

from diffusion_editor.utils.image_utils import tensor_to_images


def save_image_grid(images: Sequence[Tensor], labels: Sequence[str], path: str | Path,
                    columns: int = 3) -> None:
    """Save equal-sized single-image tensors in a labelled grid without resizing."""
    if not images or len(images) != len(labels) or type(columns) is not int or columns < 1:
        raise ValueError("grid requires images, matching labels and positive columns")
    if any(x.ndim != 4 or x.shape[0] != 1 for x in images):
        raise ValueError("each grid cell must contain a single BCHW image")
    cells = [tensor_to_images(x)[0] for x in images]
    if any(x.size != cells[0].size for x in cells):
        raise ValueError("grid images must have matching spatial dimensions")
    width, height = cells[0].size
    columns = min(columns, len(cells))
    canvas = Image.new("RGB", (width * columns, (height + 24) * math.ceil(len(cells) / columns)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (cell, label) in enumerate(zip(cells, labels)):
        x, y = index % columns * width, index // columns * (height + 24)
        canvas.paste(cell, (x, y + 24))
        draw.text((x + 8, y + 6), label, fill="black")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
