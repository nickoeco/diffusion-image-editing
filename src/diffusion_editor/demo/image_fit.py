"""Explicit SD 1.5 demo resizing/padding and aligned mask coordinate transforms."""
from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from diffusion_editor.demo.inputs import normalize_image, resolve_mask


def image_digest(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.mode}:{image.size}:".encode() + image.tobytes()).hexdigest()


@dataclass(frozen=True)
class ImageFit:
    """Sizes are width/height; padding is right/bottom, without content cropping."""
    source_size: tuple[int, int]
    content_size: tuple[int, int]
    canvas_size: tuple[int, int]
    alignment: int = 8
    max_side: int = 1024
    enabled: bool = True

    @classmethod
    def plan(cls, size: tuple[int, int], *, enabled: bool = True,
             alignment: int = 8, max_side: int = 1024) -> "ImageFit":
        if type(enabled) is not bool or type(alignment) is not int or type(max_side) is not int:
            raise ValueError("Fitting requires a boolean switch and integer alignment/limit.")
        if alignment < 1 or max_side < alignment or len(size) != 2 or any(type(x) is not int or x < 1 for x in size):
            raise ValueError("Image dimensions must be positive, with a valid alignment and size limit.")
        if not enabled:
            return cls(size, size, size, alignment, max_side, False)
        limit = max_side // alignment * alignment
        longest = max(size)
        # Integer half-up rounding makes aspect-preserving fitting deterministic.
        content = tuple(max(1, (2 * side * limit + longest) // (2 * longest)) for side in size) if longest > limit else size
        canvas = tuple((side + alignment - 1) // alignment * alignment for side in content)
        return cls(size, content, canvas, alignment, max_side, True)

    @property
    def changed(self) -> bool:
        return self.source_size != self.content_size or self.content_size != self.canvas_size

    def image(self, original: Image.Image) -> Image.Image:
        if original.mode != "RGB" or original.size != self.source_size:
            raise ValueError("Fitting requires the matching oriented RGB original.")
        content = original.resize(self.content_size, Image.Resampling.LANCZOS) if original.size != self.content_size else original.copy()
        right, bottom = (a - b for a, b in zip(self.canvas_size, self.content_size))
        if right or bottom:
            content = Image.fromarray(np.pad(np.asarray(content), ((0, bottom), (0, right), (0, 0)), mode="edge"))
        return content

    def mask(self, mask: Image.Image, *, binary: bool) -> tuple[Image.Image, dict[str, Any]]:
        if not isinstance(mask, Image.Image) or mask.mode not in ("L", "1"):
            raise ValueError("Uploaded masks must be grayscale (white edits, black preserves).")
        mask = ImageOps.exif_transpose(mask).convert("L")
        if mask.size == self.source_size:
            space = "original_upload"
            interpolation = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
            content = mask.resize(self.content_size, interpolation) if mask.size != self.content_size else mask
        elif mask.size == self.canvas_size:
            space = "model_canvas"
            content = mask.crop((0, 0, *self.content_size))
        elif mask.size == self.content_size:
            space = "resized_content"
            content = mask
        else:
            raise ValueError(f"Mask size {mask.size} is not aligned. Use original upload {self.source_size}, "
                             f"resized content {self.content_size}, or shown canvas {self.canvas_size}.")
        fitted = Image.new("L", self.canvas_size, 0)
        fitted.paste(content, (0, 0))
        return fitted, {"input_size": mask.size, "coordinate_space": space,
                        "resize_filter": ("nearest" if binary else "bilinear") if mask.size != content.size and space == "original_upload" else "none",
                        "padding_weight": 0}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"padding_right": self.canvas_size[0] - self.content_size[0],
                              "padding_bottom": self.canvas_size[1] - self.content_size[1],
                              "image_resize_filter": "lanczos" if self.source_size != self.content_size else "none",
                              "image_padding": "replicate_edge", "output_space": "fitted_model_canvas"}

    def summary(self) -> str:
        label = "Automatic fitting" if self.enabled else "Fitting disabled"
        sizes = [f"{w} x {h}" for w, h in (self.source_size, self.content_size, self.canvas_size)]
        return (f"{label}: upload {sizes[0]} -> content {sizes[1]} -> canvas/output {sizes[2]}. "
                f"Padding right/bottom: {self.canvas_size[0] - self.content_size[0]}/{self.canvas_size[1] - self.content_size[1]} px. "
                "No crop or output upscaling.")


@dataclass(frozen=True)
class PreparedUpload:
    original: Image.Image
    image: Image.Image
    fit: ImageFit
    original_digest: str
    canvas_digest: str


def prepare_upload(image: Image.Image | None, state: PreparedUpload | None = None,
                   *, enabled: bool = True) -> PreparedUpload:
    """Reuse source pixels only when the current image matches the stored preview."""
    original = normalize_image(image)
    if isinstance(state, PreparedUpload) and image_digest(original) == state.canvas_digest:
        if state.fit.enabled == enabled:
            return state
        original = state.original
    fit = ImageFit.plan(original.size, enabled=enabled)
    canvas = fit.image(original)
    return PreparedUpload(original, canvas, fit, image_digest(original), image_digest(canvas))


def resolve_fitted_mask(prepared: PreparedUpload, source: str, uploaded: Image.Image | None,
                        drawing: dict[str, Any] | None, *, binary: bool) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Keep original/canvas masks aligned; reject unrelated-size or stale drawings."""
    if source == "Upload":
        if not isinstance(uploaded, Image.Image):
            raise ValueError("Upload a mask or choose Draw and paint an edit region.")
        raw = ImageOps.exif_transpose(uploaded).copy()
    elif source == "Draw":
        background = drawing.get("background") if isinstance(drawing, dict) else None
        if not isinstance(background, Image.Image):
            raise ValueError("Load the original image into the drawing canvas first.")
        digest = image_digest(background.convert("RGB"))
        if digest == prepared.canvas_digest:
            reference = prepared.image
        elif digest == prepared.original_digest:
            reference = prepared.original
        else:
            raise ValueError("Drawing belongs to a different image. Reload the image and redraw the mask.")
        raw = resolve_mask(reference, "Draw", None, drawing)
    else:
        raise ValueError("Mask source must be Upload or Draw.")
    mask, info = prepared.fit.mask(raw, binary=binary)
    if source == "Draw" and not mask.getbbox():
        raise ValueError("Paint inside the image content; padding is preserved.")
    return mask, raw, info
