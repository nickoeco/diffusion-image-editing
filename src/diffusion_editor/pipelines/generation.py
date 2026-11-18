"""Text-to-image orchestration using our eta=0 sampler and manual CFG."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from diffusion_editor.diffusion.sampler import DDIMSampler, SamplingResult
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.models.noise_predictor import UNetNoisePredictor
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.text_encoder import TextEncoderWrapper
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.pipelines.conditioning import encode_conditioning


@dataclass(frozen=True)
class GenerationResult:
    """RGB BCHW [0,1], clean sampling result, and initial scaled noise on CPU."""

    image: Tensor
    sampling: SamplingResult
    initial_latent: Tensor


class TextToImagePipeline:
    """Sample a Gaussian prior at the highest actual selected scheduler level.

    The prior approximates that noise level; no inversion/source image is used.
    A CPU generator fixes initial FP32 noise independently of the global RNG.
    We do not call a built-in generation pipeline or scheduler step method.
    """

    def __init__(self, backend: SDBackend) -> None:
        self.backend = backend
        self.vae = VAEWrapper(backend.vae)
        self.text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
        self.predict_noise = UNetNoisePredictor(backend.unet)

    @torch.inference_mode()
    def generate(self, prompt: str, *, negative_prompt: str = "", width: int = 512,
                 height: int = 512, num_steps: int = 50, guidance_scale: float = 7.5,
                 seed: int = 42) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a generation prompt.")
        if not isinstance(negative_prompt, str):
            raise ValueError("Negative prompt must be text.")
        factor = self.vae.spatial_factor
        if any(type(size) is not int or size < factor or size % factor for size in (width, height)):
            raise ValueError(f"Width and height must be positive integer multiples of {factor}.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32).")
        schedule = DDIMSchedule.from_scheduler(self.backend.scheduler, num_steps)
        embeddings = encode_conditioning(self.text, prompt, negative_prompt, guidance_scale, 1)
        sigma = float(self.backend.scheduler.init_noise_sigma)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Scheduler init_noise_sigma must be finite and positive.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.randn((1, self.vae.latent_channels, height // factor, width // factor),
                              generator=generator, dtype=torch.float32) * sigma
        sampling = DDIMSampler(schedule, self.predict_noise).sample(
            initial.to(self.vae.device), embeddings, guidance_scale)
        image = self.vae.decode_latents(sampling.final_latent)
        if image.shape != (1, 3, height, width):
            raise RuntimeError("Generated image dimensions do not match the requested size.")
        return GenerationResult(image, sampling, initial)
