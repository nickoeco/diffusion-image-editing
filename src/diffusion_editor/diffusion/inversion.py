"""Approximate deterministic DDIM inversion, with all update equations explicit."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from diffusion_editor.diffusion.schedule import CLEAN, DDIMSchedule
from diffusion_editor.diffusion.step import ddim_step
from diffusion_editor.diffusion.guidance import NoisePredictor, predict_guided_noise, validate_conditioning


def ddim_inversion_step(latent: Tensor, epsilon: Tensor, alpha_t: Tensor, alpha_s: Tensor) -> Tensor:
    """Move t -> higher-noise s using a fixed epsilon estimate (eta=0).

    Latent/epsilon are same-device BCHW. Coefficients and updates use FP32 even
    when the pretrained predictor uses FP16. No clipping or stochastic noise.
    """
    return ddim_step(latent, epsilon, alpha_t, alpha_s, direction="inversion")


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
    exact inverse of denoising. Embeddings may be conditional B (guidance 1 only)
    or paired 2B [unconditional, conditional] for manual CFG.
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
        if latents.ndim != 4 or min(latents.shape) < 1 or not latents.is_floating_point():
            raise ValueError("latents must be a nonempty floating BCHW tensor")
        validate_conditioning(latents, source_embeddings, guidance_scale)
        if not torch.isfinite(latents).all():
            raise ValueError("latents must be finite")
        levels = self.schedule.inversion_timesteps
        prediction_ids = []
        current = latents.detach().float().clone()
        states = {CLEAN: current.to(trajectory_device).clone()}
        # Cache the small coefficient set on the compute device once.
        coefficients = {t: self.schedule.alpha(t, device=current.device) for t in levels}
        for t, s in zip(levels, levels[1:]):
            prediction_t = s if t == CLEAN else t
            epsilon = predict_guided_noise(self.predict_noise, current, prediction_t, source_embeddings, guidance_scale)
            current = ddim_inversion_step(current, epsilon, coefficients[t], coefficients[s])
            states[s] = current.to(trajectory_device).clone()
            prediction_ids.append(prediction_t)
        return InversionResult(MappingProxyType(states), levels, tuple(prediction_ids),
                               self.schedule, float(guidance_scale))
