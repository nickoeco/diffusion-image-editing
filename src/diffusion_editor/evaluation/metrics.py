"""Whole-image RGB reconstruction metrics; no masked-region shortcuts."""
import math

import numpy as np
from torch import Tensor

from diffusion_editor.utils.image_utils import image_to_tensor


def reconstruction_metrics(reference: Tensor, reconstruction: Tensor) -> dict[str, float]:
    """Evaluate one RGB [1,3,H,W] pair in [0,1]; SSIM uses uniform 7x7 windows.

    PSNR uses RGB MSE over all pixels/channels and data range 1. Identical images
    have positive-infinite PSNR. SSIM is channel-averaged with sample covariance;
    both spatial dimensions must be at least 7. No image is zero-masked or cropped.
    """
    from skimage.metrics import structural_similarity
    if (reference.ndim != 4 or reconstruction.ndim != 4 or reference.shape[0] != 1
            or reference.shape != reconstruction.shape or min(reference.shape[-2:]) < 7):
        raise ValueError("metrics require matching single-image shapes with H,W >= 7")
    a, b = (image_to_tensor(x).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            for x in (reference, reconstruction))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = math.inf if mse == 0 else -10 * math.log10(mse)
    ssim = structural_similarity(a, b, data_range=1.0, channel_axis=-1, win_size=7,
                                 gaussian_weights=False, use_sample_covariance=True)
    if not math.isfinite(ssim):
        raise RuntimeError("SSIM produced a nonfinite value")
    return {"psnr_db": psnr, "ssim": float(ssim)}


def background_metrics(reference: Tensor, edited: Tensor, pixel_mask: Tensor) -> dict[str, float | int | None]:
    """One RGB image: soft background weights (1-M), plus literal zero-mask pixels.

    MAE/MSE denominators are 3*sum(weights), not the whole image area. An empty
    region returns None. PSNR is +inf for a defined, exactly preserved background.
    No perceptual metric is computed on artificially zeroed images.
    """
    from diffusion_editor.editing.mask import validate_mask
    if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape != edited.shape:
        raise ValueError("background metrics require matching single-image BCHW tensors")
    validate_mask(pixel_mask, 1, *reference.shape[-2:])
    a, b = (image_to_tensor(x).detach().cpu().double() for x in (reference, edited))
    mask = pixel_mask.detach().cpu().double()
    weights = 1 - mask
    weight = float(weights.sum())
    zero = mask == 0
    zero_count = int(zero.sum())
    error = (a - b).abs()
    mae = float((error * weights).sum() / (3 * weight)) if weight else None
    mse = float((error.square() * weights).sum() / (3 * weight)) if weight else None
    psnr = (math.inf if mse == 0 else -10 * math.log10(mse)) if mse is not None else None
    zero_mae = float((error * zero).sum() / (3 * zero_count)) if zero_count else None
    return {"background_weight_pixels": weight, "background_mae": mae, "background_psnr_db": psnr,
            "zero_mask_pixels": zero_count, "zero_mask_mae": zero_mae}
