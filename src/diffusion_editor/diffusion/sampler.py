"""Custom deterministic reverse DDIM sampling on the inversion schedule."""
from dataclasses import dataclass
from collections.abc import Callable
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from diffusion_editor.diffusion.guidance import NoisePredictor, predict_guided_noise, validate_conditioning
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.diffusion.step import ddim_step


def ddim_sampling_step(latent: Tensor, epsilon: Tensor, alpha_t: Tensor, alpha_s: Tensor) -> Tensor:
    """Move t -> lower-noise s, including s=CLEAN with alpha_s=1, at eta=0."""
    return ddim_step(latent, epsilon, alpha_t, alpha_s, direction="sampling")


@dataclass(frozen=True)
class SamplingResult:
    final_latent: Tensor
    timesteps: tuple[int, ...]
    trajectory: Mapping[int, Tensor] | None


class DDIMSampler:
    """Query each current noisy timestep and traverse exactly the shared grid."""

    def __init__(self, schedule: DDIMSchedule, predict_noise: NoisePredictor) -> None:
        self.schedule = schedule
        self.predict_noise = predict_noise

    @torch.inference_mode()
    def sample(self, initial_latent: Tensor, prompt_embeddings: Tensor, guidance_scale: float = 1.0,
               *, return_trajectory: bool = False,
               trajectory_device: str | torch.device = "cpu",
               start_timestep: int | None = None,
               after_step: Callable[[Tensor, int], Tensor] | None = None) -> SamplingResult:
        """Start at a selected level (default highest); return a clean FP32 latent.

        The caller must supply the latent at start_timestep from a matching
        inversion trajectory. CLEAN is a no-update path. No new noise is added.
        An optional callback receives the UPDATED latent and its destination ID,
        including CLEAN. Its FP32 output is validated before snapshot storage.
        """
        if initial_latent.ndim != 4 or min(initial_latent.shape) < 1 or not initial_latent.is_floating_point():
            raise ValueError("initial latent must be nonempty floating BCHW")
        if not torch.isfinite(initial_latent).all():
            raise ValueError("initial latent must be finite")
        validate_conditioning(initial_latent, prompt_embeddings, guidance_scale)
        current = initial_latent.detach().float().clone()
        all_levels = self.schedule.reverse_timesteps
        start = all_levels[0] if start_timestep is None else start_timestep
        if type(start) is not int or start not in all_levels:
            raise ValueError("start_timestep must be an explicit level in the shared schedule")
        levels = all_levels[all_levels.index(start):]
        states = {levels[0]: current.to(trajectory_device).clone()} if return_trajectory else None
        coefficients = {t: self.schedule.alpha(t, device=current.device) for t in levels}
        for t, s in zip(levels, levels[1:]):
            epsilon = predict_guided_noise(self.predict_noise, current, t, prompt_embeddings, guidance_scale)
            current = ddim_sampling_step(current, epsilon, coefficients[t], coefficients[s])
            if after_step is not None:
                updated = after_step(current, s)
                if (not isinstance(updated, Tensor) or updated.shape != current.shape
                        or updated.device != current.device or updated.dtype != torch.float32
                        or not torch.isfinite(updated).all()):
                    raise ValueError("after_step must return finite FP32 latents with unchanged shape/device")
                current = updated
            if states is not None:
                states[s] = current.to(trajectory_device).clone()
        return SamplingResult(current, levels, MappingProxyType(states) if states is not None else None)
