"""Approximate deterministic DDIM inversion, with all update equations explicit."""
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from diffusion_editor.diffusion.schedule import CLEAN, DDIMSchedule

NoisePredictor = Callable[[Tensor, int, Tensor], Tensor]


def ddim_inversion_step(latent: Tensor, epsilon: Tensor, alpha_t: Tensor, alpha_s: Tensor) -> Tensor:
    """Move t -> higher-noise s using a fixed epsilon estimate (eta=0).

    Latent/epsilon are same-device BCHW. Coefficients and updates use FP32 even
    when the pretrained predictor uses FP16. No clipping or stochastic noise.
    """
    if latent.ndim != 4 or min(latent.shape) < 1 or epsilon.shape != latent.shape:
        raise ValueError("latent and epsilon must have matching nonempty BCHW shapes")
    if not latent.is_floating_point() or not epsilon.is_floating_point():
        raise ValueError("latent and epsilon must be floating point")
    if latent.device != epsilon.device:
        raise ValueError("latent and epsilon must be on the same device")
    t, s = (torch.as_tensor(a, device=latent.device, dtype=torch.float32) for a in (alpha_t, alpha_s))
    if t.ndim != 0 or s.ndim != 0 or not (0 < s < t <= 1):
        raise ValueError("inversion requires scalar coefficients 0 < alpha_s < alpha_t <= 1")
    if not torch.isfinite(latent).all() or not torch.isfinite(epsilon).all():
        raise ValueError("latent and epsilon must be finite")
    z, eps = latent.float(), epsilon.float()
    clean_estimate = (z - (1 - t).sqrt() * eps) / t.sqrt()
    result = s.sqrt() * clean_estimate + (1 - s).sqrt() * eps
    if not torch.isfinite(result).all():
        raise RuntimeError("DDIM inversion step produced nonfinite values")
    return result


@dataclass(frozen=True)
class InversionResult:
    """FP32 snapshots keyed by explicit timestep ID, including the clean state."""
    trajectory: Mapping[int, Tensor]
    timesteps: tuple[int, ...]
    prediction_timesteps: tuple[int, ...]
    schedule: DDIMSchedule
    guidance_scale: float

    @property
    def initial_latent(self) -> Tensor:
        return self.trajectory[CLEAN]

    @property
    def final_latent(self) -> Tensor:
        return self.trajectory[self.timesteps[-1]]


class DDIMInverter:
    """Invert a clean latent along a fixed schedule using source conditioning.

    Predictor is called on the CURRENT state at its own training timestep. At
    CLEAN only, use the lowest selected training timestep as a proxy: -1 is a
    bookkeeping ID and is never sent to the U-Net. This approximation is not an
    exact inverse of denoising. Guidance 1 (conditional-only) is supported here;
    manual CFG arithmetic is intentionally deferred to Phase 3.
    """

    def __init__(self, schedule: DDIMSchedule, predict_noise: NoisePredictor) -> None:
        self.schedule = schedule
        self.predict_noise = predict_noise

    @torch.inference_mode()
    def invert(
        self, latents: Tensor, source_embeddings: Tensor, guidance_scale: float = 1.0,
        *, trajectory_device: str | torch.device = "cpu",
    ) -> InversionResult:
        """Return N+1 snapshots; CPU storage avoids keeping a full GPU trajectory."""
        if guidance_scale != 1.0:
            raise ValueError("Phase 2 supports guidance_scale=1 only; CFG is a Phase 3 deliverable")
        if latents.ndim != 4 or min(latents.shape) < 1 or not latents.is_floating_point():
            raise ValueError("latents must be a nonempty floating BCHW tensor")
        if source_embeddings.ndim != 3 or source_embeddings.shape[0] != latents.shape[0]:
            raise ValueError("source embeddings must be [B,L,D] with the latent batch size")
        if not source_embeddings.is_floating_point() or min(source_embeddings.shape) < 1:
            raise ValueError("source embeddings must be nonempty floating values")
        if not torch.isfinite(latents).all() or not torch.isfinite(source_embeddings).all():
            raise ValueError("latents and source embeddings must be finite")
        levels = self.schedule.inversion_timesteps
        prediction_ids = []
        current = latents.detach().float().clone()
        states = {CLEAN: current.to(trajectory_device).clone()}
        # Cache the small coefficient set on the compute device once.
        coefficients = {t: self.schedule.alpha(t, device=current.device) for t in levels}
        for t, s in zip(levels, levels[1:]):
            prediction_t = s if t == CLEAN else t
            epsilon = self.predict_noise(current, prediction_t, source_embeddings)
            current = ddim_inversion_step(current, epsilon, coefficients[t], coefficients[s])
            states[s] = current.to(trajectory_device).clone()
            prediction_ids.append(prediction_t)
        return InversionResult(MappingProxyType(states), levels, tuple(prediction_ids),
                               self.schedule, float(guidance_scale))
