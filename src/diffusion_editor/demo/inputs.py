"""ImageEditor layer extraction without using the photographic composite as a mask."""
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def normalize_image(image: Image.Image | None) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("Upload an original image first.")
    return ImageOps.exif_transpose(image).convert("RGB")


def editor_canvas(image: Image.Image | None) -> dict[str, Any] | None:
    """Reset painting whenever the original changes; preserve original dimensions."""
    if image is None:
        return None
    original = normalize_image(image).convert("RGBA")
    return {"background": original, "layers": [], "composite": original.copy()}


def resolve_mask(original: Image.Image, source: str, uploaded: Image.Image | None,
                  drawing: dict[str, Any] | None) -> Image.Image:
    """Upload uses explicit grayscale weights; Draw uses only layer alpha union."""
    if source == "Upload":
        if not isinstance(uploaded, Image.Image):
            raise ValueError("Upload a mask or choose Draw and paint an edit region.")
        if uploaded.mode not in ("L", "1"):
            raise ValueError("Uploaded masks must be grayscale (white edits, black preserves).")
        mask = ImageOps.exif_transpose(uploaded).convert("L")
    elif source == "Draw":
        if not isinstance(drawing, dict) or not isinstance(drawing.get("background"), Image.Image):
            raise ValueError("Load the original image into the drawing canvas first.")
        background = drawing["background"].convert("RGB")
        if background.size != original.size or not np.array_equal(np.asarray(background), np.asarray(original)):
            raise ValueError("Drawing belongs to a different image. Reload the image and redraw the mask.")
        alpha = np.zeros((original.height, original.width), dtype=np.float32)
        for layer in drawing.get("layers") or []:
            if not isinstance(layer, Image.Image) or layer.mode != "RGBA" or layer.size != original.size:
                raise ValueError("Paint layers must be RGBA and match the original image size.")
            layer_alpha = np.asarray(layer, dtype=np.float32)[..., 3] / 255
            alpha = 1 - (1 - alpha) * (1 - layer_alpha)
        if not np.any(alpha > 0):
            raise ValueError("No painted region. Paint over the area to edit or upload a mask.")
        mask = Image.fromarray(np.rint(alpha * 255).astype(np.uint8))
    else:
        raise ValueError("Mask source must be Upload or Draw.")
    if mask.size != original.size:
        raise ValueError(f"Mask size {mask.size} does not match image {original.size}; upload an aligned mask.")
    return mask
