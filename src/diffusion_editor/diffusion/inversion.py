"""Approximate deterministic DDIM inversion, with all update equations explicit."""
from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from diffusion_editor.diffusion.schedule import CLEAN, DDIMSchedule
from diffusion_editor.diffusion.step import ddim_step
from diffusion_editor.diffusion.guidance import NoisePredictor, predict_guided_noise, validate_conditioning
from diffusion_editor.diffusion.refinement import RefinementDiagnostics, refine_inversion_step, validate_refinement

logger = logging.getLogger(__name__)


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
    refinement: Mapping[int, RefinementDiagnostics] | None = None

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
    Optional residual refinement evaluates the destination network to numerically
    invert the actual reverse step; zero retains the original one-prediction path.
    """

    def __init__(self, schedule: DDIMSchedule, predict_noise: NoisePredictor) -> None:
        self.schedule = schedule
        self.predict_noise = predict_noise

    @torch.inference_mode()
    def invert(
        self, latents: Tensor, source_embeddings: Tensor, guidance_scale: float = 1.0,
        *, trajectory_device: str | torch.device = "cpu", refinement_steps: int = 0,
        refinement_tolerance: float = 1e-6,
    ) -> InversionResult:
        """Return N+1 snapshots and optional per-level refinement diagnostics.

        prediction_timesteps records initial estimates; additional destination
        queries are counted by refinement[t].evaluations. CPU storage limits VRAM.
        """
        if latents.ndim != 4 or min(latents.shape) < 1 or not latents.is_floating_point():
            raise ValueError("latents must be a nonempty floating BCHW tensor")
        validate_conditioning(latents, source_embeddings, guidance_scale)
        validate_refinement(refinement_steps, refinement_tolerance)
        if not torch.isfinite(latents).all():
            raise ValueError("latents must be finite")
        levels = self.schedule.inversion_timesteps
        prediction_ids = []
        current = latents.detach().float().clone()
        states = {CLEAN: current.to(trajectory_device).clone()}
        refinement = {}
        # Cache the small coefficient set on the compute device once.
        coefficients = {t: self.schedule.alpha(t, device=current.device) for t in levels}
        for t, s in zip(levels, levels[1:]):
            prediction_t = s if t == CLEAN else t
            epsilon = predict_guided_noise(self.predict_noise, current, prediction_t, source_embeddings, guidance_scale)
            candidate = ddim_inversion_step(current, epsilon, coefficients[t], coefficients[s])
            if refinement_steps:
                candidate, diagnostics = refine_inversion_step(
                    current, candidate, coefficients[t], coefficients[s], s,
                    self.predict_noise, source_embeddings, guidance_scale,
                    max_iterations=refinement_steps, tolerance=refinement_tolerance)
                refinement[s] = diagnostics
            current = candidate
            states[s] = current.to(trajectory_device).clone()
            prediction_ids.append(prediction_t)
        unconverged = [d for d in refinement.values() if not d.converged]
        if unconverged:
            logger.warning("Refinement tolerance not reached at %d/%d levels; returning best evaluated states (max RMS %.6g)",
                           len(unconverged), len(refinement), max(d.final_rms for d in unconverged))
        return InversionResult(MappingProxyType(states), levels, tuple(prediction_ids),
                               self.schedule, float(guidance_scale),
                               MappingProxyType(refinement) if refinement_steps else None)
