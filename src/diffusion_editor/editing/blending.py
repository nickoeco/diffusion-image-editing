"""Mask-weighted latent blending and independently configurable pixel compositing."""
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from diffusion_editor.editing.mask import validate_mask


def blend_latents(edited: Tensor, source: Tensor, mask: Tensor) -> Tensor:
    """Compute M*edited+(1-M)*source at the SAME noise level, in FP32."""
    if edited.ndim != 4 or min(edited.shape) < 1 or source.shape != edited.shape:
        raise ValueError("source and edited tensors must have matching nonempty BCHW shapes")
    if not edited.is_floating_point() or not source.is_floating_point():
        raise ValueError("source and edited tensors must be floating point")
    if edited.device != source.device or mask.device != edited.device:
        raise ValueError("blend tensors and mask must share a device")
    validate_mask(mask, edited.shape[0], *edited.shape[-2:])
    if not torch.isfinite(edited).all() or not torch.isfinite(source).all():
        raise ValueError("blend tensors must be finite")
    m = mask.float()
    return m * edited.float() + (1 - m) * source.float()


def composite_pixels(edited: Tensor, original: Tensor, pixel_mask: Tensor) -> Tensor:
    """Use the pixel mask independently of latent blending; zero weights preserve exactly."""
    return blend_latents(edited, original.to(device=edited.device, dtype=torch.float32),
                         pixel_mask.to(device=edited.device, dtype=torch.float32))


class SourceLatentBlender:
    """Sampler after-step callback: look up the UPDATED state ID, including clean."""

    def __init__(self, source_trajectory: Mapping[int, Tensor], latent_mask: Tensor,
                 required_timesteps: Sequence[int], reference_latent: Tensor) -> None:
        validate_mask(latent_mask, reference_latent.shape[0], *reference_latent.shape[-2:])
        self.mask = latent_mask.to(device=reference_latent.device, dtype=torch.float32)
        self.source = source_trajectory
        self.required = frozenset(required_timesteps)
        for timestep in self.required:
            if timestep not in source_trajectory:
                raise ValueError(f"source trajectory missing updated timestep {timestep}")
            state = source_trajectory[timestep]
            if state.shape != reference_latent.shape or not state.is_floating_point() or not torch.isfinite(state).all():
                raise ValueError(f"invalid source latent at timestep {timestep}")

    def __call__(self, updated: Tensor, timestep: int) -> Tensor:
        if timestep not in self.required:
            raise ValueError(f"unexpected updated timestep {timestep}")
        source = self.source[timestep].to(device=updated.device, dtype=torch.float32)
        return blend_latents(updated, source, self.mask)
