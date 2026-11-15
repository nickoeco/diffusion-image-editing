"""Manual classifier-free guidance, separate from DDIM update equations."""
from collections.abc import Callable
import math
from numbers import Real

import torch
from torch import Tensor

NoisePredictor = Callable[[Tensor, int, Tensor], Tensor]


def validate_guidance_scale(scale: float) -> None:
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance scale must be finite and nonnegative")


def classifier_free_guidance(unconditional: Tensor, conditional: Tensor, scale: float) -> Tensor:
    """Compute u + w*(c-u) in FP32; 0 selects u and 1 selects c."""
    validate_guidance_scale(scale)
    if unconditional.shape != conditional.shape or unconditional.ndim != 4 or min(unconditional.shape) < 1:
        raise ValueError("CFG predictions must have matching nonempty BCHW shapes")
    if unconditional.device != conditional.device:
        raise ValueError("CFG predictions must be on the same device")
    if not unconditional.is_floating_point() or not conditional.is_floating_point():
        raise ValueError("CFG predictions must be floating point")
    if not torch.isfinite(unconditional).all() or not torch.isfinite(conditional).all():
        raise ValueError("CFG predictions must be finite")
    u, c = unconditional.float(), conditional.float()
    # Preserve the exact endpoint tensors, avoiding cancellation at scale 1.
    result = u if scale == 0 else c if scale == 1 else u + scale * (c - u)
    if not torch.isfinite(result).all():
        raise RuntimeError("CFG produced nonfinite values")
    return result


def validate_conditioning(latents: Tensor, embeddings: Tensor, scale: float) -> None:
    """B embeddings mean conditional-only at w=1; 2B mean [uncond, cond]."""
    validate_guidance_scale(scale)
    if embeddings.ndim != 3 or min(embeddings.shape) < 1 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be nonempty floating [B,L,D] or [2B,L,D]")
    batch = latents.shape[0]
    if embeddings.shape[0] not in (batch, 2 * batch):
        raise ValueError("embedding batch must be B or 2B with [unconditional, conditional] order")
    if embeddings.shape[0] == batch and scale != 1:
        raise ValueError("guidance scales other than 1 require 2B unconditional/conditional embeddings")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")


def predict_guided_noise(predictor: NoisePredictor, latents: Tensor, timestep: int,
                         embeddings: Tensor, scale: float) -> Tensor:
    """One model call; duplicate latent batch only for paired conditioning."""
    validate_conditioning(latents, embeddings, scale)
    if embeddings.shape[0] == latents.shape[0]:
        return predictor(latents, timestep, embeddings)
    doubled = torch.cat((latents, latents), dim=0)
    predictions = predictor(doubled, timestep, embeddings)
    if predictions.shape != doubled.shape:
        raise ValueError("batched CFG predictor returned an invalid shape")
    unconditional, conditional = predictions.chunk(2, dim=0)
    return classifier_free_guidance(unconditional, conditional, scale)
