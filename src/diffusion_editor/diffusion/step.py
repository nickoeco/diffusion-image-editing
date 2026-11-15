"""Shared explicit eta=0 DDIM update with direction-aware coefficient checks."""
from typing import Literal

import torch
from torch import Tensor


def ddim_step(latent: Tensor, epsilon: Tensor, alpha_t: Tensor, alpha_s: Tensor,
              *, direction: Literal["inversion", "sampling"]) -> Tensor:
    """Estimate x0 then transfer to level s, always using FP32 arithmetic."""
    if latent.ndim != 4 or min(latent.shape) < 1 or epsilon.shape != latent.shape:
        raise ValueError("latent and epsilon must have matching nonempty BCHW shapes")
    if not latent.is_floating_point() or not epsilon.is_floating_point():
        raise ValueError("latent and epsilon must be floating point")
    if latent.device != epsilon.device:
        raise ValueError("latent and epsilon must be on the same device")
    t, s = (torch.as_tensor(a, device=latent.device, dtype=torch.float32) for a in (alpha_t, alpha_s))
    if direction not in ("inversion", "sampling"):
        raise ValueError("invalid DDIM direction")
    valid = t.ndim == s.ndim == 0
    if valid:
        valid = bool(0 < s < t <= 1) if direction == "inversion" else bool(0 < t < s <= 1)
    if not valid:
        raise ValueError(f"{direction} requires scalar coefficients with strictly ordered noise levels in (0,1]")
    if not torch.isfinite(latent).all() or not torch.isfinite(epsilon).all():
        raise ValueError("latent and epsilon must be finite")
    z, eps = latent.float(), epsilon.float()
    clean_estimate = (z - (1 - t).sqrt() * eps) / t.sqrt()
    result = s.sqrt() * clean_estimate + (1 - s).sqrt() * eps
    if not torch.isfinite(result).all():
        raise RuntimeError("DDIM step produced nonfinite values")
    return result
