"""Budgeted residual correction for inversion of the actual reverse DDIM step."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from diffusion_editor.diffusion.guidance import NoisePredictor, predict_guided_noise
from diffusion_editor.diffusion.step import ddim_step


@dataclass(frozen=True)
class RefinementDiagnostics:
    initial_rms: float
    final_rms: float
    evaluations: int
    converged: bool


def validate_refinement(max_iterations: int, tolerance: float) -> None:
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("refinement iterations must be a nonnegative integer")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("refinement tolerance must be finite and positive")


@torch.inference_mode()
def refine_inversion_step(source: Tensor, initial: Tensor, alpha_t: Tensor, alpha_s: Tensor,
                           destination_timestep: int, predictor: NoisePredictor, embeddings: Tensor,
                           guidance_scale: float, *, max_iterations: int = 12,
                           tolerance: float = 1e-6) -> tuple[Tensor, RefinementDiagnostics]:
    """Reduce R_s(candidate)-source, keeping the best evaluated candidate.

    R_s is the SAME eta=0 reverse step used by sampling. For R_s(z)=a*z+b*eps(z,s),
    a=sqrt(alpha_t/alpha_s), the correction z <- z - (R_s(z)-source)/a is a
    fixed-point update. Rejected trials halve damping and restart from the best
    point. No weights/embeddings are trained; all work is inference-only FP32.

    The budget counts destination U-Net evaluations, including the initial
    residual check. No convergence or global-error guarantee is made. The last
    unevaluated proposal is never returned. RMS is over the full BCHW tensor.
    """
    validate_refinement(max_iterations, tolerance)
    if max_iterations == 0:
        raise ValueError("refine_inversion_step requires at least one evaluation")
    if source.shape != initial.shape or source.device != initial.device:
        raise ValueError("source and initial candidate must match shape and device")
    if not source.is_floating_point() or not initial.is_floating_point():
        raise ValueError("source and initial candidate must be floating point")
    if type(destination_timestep) is not int or destination_timestep < 0:
        raise ValueError("destination timestep must be nonnegative")
    t, s = (torch.as_tensor(a, device=source.device, dtype=torch.float32) for a in (alpha_t, alpha_s))
    if t.ndim != 0 or s.ndim != 0 or not (0 < s < t <= 1):
        raise ValueError("refinement requires inversion coefficients 0 < alpha_s < alpha_t <= 1")
    if not torch.isfinite(source).all():
        raise ValueError("source must be finite")
    source = source.float()
    candidate = initial.detach().float().clone()
    best, best_rms, best_residual = candidate, math.inf, None
    initial_rms = math.inf
    gain = (t / s).sqrt()
    damping = 1.0
    for evaluation in range(1, max_iterations + 1):
        epsilon = predict_guided_noise(predictor, candidate, destination_timestep, embeddings, guidance_scale)
        recovered = ddim_step(candidate, epsilon, s, t, direction="sampling")
        residual = recovered - source
        rms = float(residual.square().mean().sqrt())
        if not math.isfinite(rms):
            raise RuntimeError("inversion residual became nonfinite")
        if evaluation == 1:
            initial_rms = rms
        if rms < best_rms:
            best, best_rms, best_residual = candidate, rms, residual
        else:
            damping *= 0.5
        if best_rms <= tolerance:
            break
        candidate = best - damping * best_residual / gain
    return best, RefinementDiagnostics(initial_rms, best_rms, evaluation, best_rms <= tolerance)
