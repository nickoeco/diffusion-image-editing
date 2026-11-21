"""Explicit forward marginals and floor-based Img2Img strength selection."""
import torch
from torch import Tensor


def forward_noise(clean: Tensor, noise: Tensor, alpha: Tensor) -> Tensor:
    """q(z_t|z_0): sqrt(alpha)*z_0 + sqrt(1-alpha)*epsilon, BCHW FP32.

    alpha=1 is the exact clean boundary. Reusing epsilon gives coupled forward
    marginals, not an independently simulated Markov forward trajectory.
    """
    if (clean.ndim != 4 or min(clean.shape) < 1 or noise.shape != clean.shape
            or clean.device != noise.device or not clean.is_floating_point() or not noise.is_floating_point()):
        raise ValueError("clean/noise must be matching floating BCHW tensors on the same device")
    a = torch.as_tensor(alpha, device=clean.device, dtype=torch.float32)
    if a.ndim != 0 or not torch.isfinite(a) or not 0 < a <= 1:
        raise ValueError("alpha must be a finite scalar in (0,1]")
    if not torch.isfinite(clean).all() or not torch.isfinite(noise).all():
        raise ValueError("clean/noise must be finite")
    result = a.sqrt() * clean.float() + (1 - a).sqrt() * noise.float()
    if not torch.isfinite(result).all():
        raise RuntimeError("Forward noising produced nonfinite latents")
    return result
