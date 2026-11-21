"""Model-free geometry, overlays and immutable result associations for the demo."""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from diffusion_editor.demo.image_fit import image_digest


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def planned_reverse_steps(method: str, steps: Any, strength: Any) -> int:
    """Display the same decimal floor/ceil convention as the core selectors."""
    count = integer(steps, "Steps")
    if not 1 <= count <= 200 or isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Use 1–200 steps and strength in [0,1].")
    product = Decimal(str(float(strength))) * count
    if method == "Forward noising":
        return math.floor(product)
    if method == "DDIM inversion":
        return math.ceil(product)
    raise ValueError("Choose Forward noising or DDIM inversion.")


def frozen_settings(value: Any) -> Any:
    """Serializable fingerprints can replace images without changing their digest."""
    if isinstance(value, Image.Image):
        return {"pixels": image_digest(value)}
    if isinstance(value, dict):
        if "background" in value and "layers" in value:
            # ImageEditor may insert an empty transparent layer or re-export its
            # photographic composite. Only RGB background and layer-alpha union
            # enter mask resolution; fingerprint those exact semantics.
            background = value["background"]
            if not isinstance(background, Image.Image):
                raise ValueError("The drawing canvas has no image background.")
            alpha = np.zeros((background.height, background.width), dtype=np.float32)
            for layer in value.get("layers") or []:
                if not isinstance(layer, Image.Image) or layer.mode != "RGBA" or layer.size != background.size:
                    raise ValueError("Paint layers must be RGBA and match the canvas.")
                alpha = 1 - (1 - alpha) * (1 - np.asarray(layer, dtype=np.float32)[..., 3] / 255)
            weights = Image.fromarray(np.rint(alpha * 255).astype(np.uint8))
            return {"paint_background": image_digest(background.convert("RGB")), "paint_alpha": image_digest(weights)}
        return {key: frozen_settings(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [frozen_settings(item) for item in value]
    # PreparedUpload deliberately records both source and canvas hashes.
    if hasattr(value, "canvas_digest"):
        return {"source": value.original_digest, "canvas": value.canvas_digest, "fit": value.fit.to_dict()}
    return value


def settings_digest(values: Any) -> str:
    """Hash values, including paint alpha/background, without retaining live objects."""
    return hashlib.sha256(json.dumps(frozen_settings(values), sort_keys=True, allow_nan=False).encode()).hexdigest()


def result_badge(snapshot: dict | None, current_digest: str) -> str:
    if not snapshot:
        return "No completed run yet."
    state = "Settings changed — showing previous result" if snapshot["settings_digest"] != current_digest else "Result matches current settings"
    return f"**{state}** · Run `{snapshot['run_id']}`"


def mask_overlay(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Cyan tint represents the requested mask, never a predicted effective mask."""
    if mask is None:
        return image.copy()
    if image.size != mask.size:
        raise ValueError("Mask preview must match the prepared canvas.")
    color = Image.new("RGB", image.size, (24, 185, 211))
    alpha = mask.convert("L").point(lambda x: round(x * 0.48))
    return Image.composite(color, image.convert("RGB"), alpha)


def outpaint_preview(image: Image.Image, left: Any, right: Any, top: Any, bottom: Any,
                     overlap: Any, fill: str) -> tuple[Image.Image, Image.Image, str]:
    """PIL/NumPy preview matching canvas.py; never resize the source rectangle."""
    left, right, top, bottom, overlap = [integer(v, "Margins / overlap") for v in (left, right, top, bottom, overlap)]
    if min(left, right, top, bottom, overlap) < 0 or not any((left, right, top, bottom)):
        raise ValueError("Use nonnegative margins and expand at least one side.")
    width, height = image.size
    if width + left + right > 1024 or height + top + bottom > 1024:
        raise ValueError("Final canvas must be at most 1024 pixels per side.")
    if (width + left + right) % 8 or (height + top + bottom) % 8 or min(image.size) < 8:
        raise ValueError("Final canvas sides must be multiples of 8; choose explicit image fitting or adjust margins.")
    if 2 * overlap >= min(image.size) or fill not in ("edge", "neutral"):
        raise ValueError("Overlap must be less than half the smaller source side; choose edge or neutral fill.")
    pixels = np.asarray(image.convert("RGB"))
    padding = ((top, bottom), (left, right), (0, 0))
    canvas = np.pad(pixels, padding, mode="edge") if fill == "edge" else np.pad(pixels, padding, constant_values=128)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    interior = np.zeros((height, width), dtype=np.float32)
    if overlap:
        for margin, distance in ((left, x), (right, width - 1 - x), (top, y), (bottom, height - 1 - y)):
            if margin:
                interior = np.maximum(interior, np.clip(1 - distance / overlap, 0, 1))
    weights = np.ones(canvas.shape[:2], dtype=np.float32)
    weights[top:top + height, left:left + width] = interior
    mask = Image.fromarray(np.rint(weights * 255).astype(np.uint8))
    box = (left, top, left + width, top + height)
    summary = (f"Canvas {canvas.shape[1]} × {canvas.shape[0]} px · source box {box}. "
               f"Cyan: extension; cyan inside the source: editable overlap ({overlap} px). "
               "Source size is unchanged. Feathering may expand the effective mask; inspect it after running.")
    return Image.fromarray(canvas), mask, summary


def boundary_comparison(before: Image.Image, after: Image.Image, mask: Image.Image | None,
                        side: int = 160) -> tuple[tuple[Image.Image, Image.Image] | None, str]:
    """Choose a fixed requested-mask contour crop, independent of the output pixels."""
    if mask is None or before.size != after.size or mask.size != before.size:
        return None, "No spatial mask boundary for this run."
    binary = mask.convert("L").point(lambda x: 255 if x > 0 else 0)
    edge = np.asarray(binary.filter(ImageFilter.MaxFilter(3))) != np.asarray(binary.filter(ImageFilter.MinFilter(3)))
    coordinates = np.argwhere(edge)
    if not len(coordinates):
        return None, "The requested mask has no interior boundary."
    y, x = coordinates[len(coordinates) // 2]
    width, height = before.size
    crop_w, crop_h = min(side, width), min(side, height)
    left, top = min(max(0, int(x) - crop_w // 2), width - crop_w), min(max(0, int(y) - crop_h // 2), height - crop_h)
    box = (left, top, left + crop_w, top + crop_h)
    return (before.crop(box), after.crop(box)), f"Requested-mask boundary crop {box} in the run's {width} × {height} canvas. Left: input; right: result."
