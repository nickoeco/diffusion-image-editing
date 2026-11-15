"""Deterministic VAE encoding with explicit normalization and latent scaling."""
import math
from typing import Any

from PIL import Image
import torch
from torch import Tensor

from diffusion_editor.utils.image_utils import image_to_tensor


class VAEWrapper:
    """Wrap the backend's FP32 VAE; no resizing, cropping or posterior sampling.

    Input/output images are RGB BCHW [0,1]. Scaled latents are BCHW FP32 on
    the VAE device; spatial compression and channel count come from its config.
    """

    def __init__(self, vae: Any) -> None:
        self.vae = vae
        self.scaling_factor = float(vae.config.scaling_factor)
        if not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0:
            raise ValueError("VAE scaling_factor must be finite and positive")
        self.spatial_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        if self.spatial_factor < 1:
            raise ValueError("VAE block_out_channels must not be empty")
        self.latent_channels = vae.config.latent_channels
        if next(vae.parameters()).dtype != torch.float32:
            raise ValueError("VAEWrapper requires an FP32 VAE; use SDBackend.load")
        self.vae.eval().requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @torch.inference_mode()
    def encode_image(self, image: Image.Image | Tensor) -> Tensor:
        """Compute z = posterior.mode(2*image-1) * config.scaling_factor."""
        pixels = image_to_tensor(image).to(device=self.device, dtype=torch.float32)
        height, width = pixels.shape[-2:]
        if height % self.spatial_factor or width % self.spatial_factor:
            raise ValueError(f"image height/width must be multiples of {self.spatial_factor}; resize explicitly")
        latent = self.vae.encode(2 * pixels - 1).latent_dist.mode() * self.scaling_factor
        expected = (pixels.shape[0], self.latent_channels,
                    height // self.spatial_factor, width // self.spatial_factor)
        if latent.shape != expected or not torch.isfinite(latent).all():
            raise RuntimeError("VAE encoder produced invalid latent shape or nonfinite values")
        return latent

    @torch.inference_mode()
    def decode_latents(self, latents: Tensor) -> Tensor:
        """Undo latent scaling, decode, map [-1,1] to [0,1] and clamp."""
        if not isinstance(latents, Tensor):
            raise TypeError("latents must be a floating BCHW tensor")
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels or min(latents.shape) < 1:
            raise ValueError("latents must have nonempty BCHW shape with VAE latent_channels")
        if not latents.is_floating_point() or not torch.isfinite(latents).all():
            raise ValueError("latents must be finite floating point values")
        latent = latents.to(device=self.device, dtype=torch.float32)
        decoded = self.vae.decode(latent / self.scaling_factor).sample
        expected = (latent.shape[0], 3, latent.shape[2] * self.spatial_factor,
                    latent.shape[3] * self.spatial_factor)
        if decoded.shape != expected or not torch.isfinite(decoded).all():
            raise RuntimeError("VAE decoder produced invalid image shape or nonfinite values")
        return (decoded / 2 + 0.5).clamp(0, 1)
