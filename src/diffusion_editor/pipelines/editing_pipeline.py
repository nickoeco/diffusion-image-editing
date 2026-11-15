"""Global source/target text editing using stored DDIM inversion levels."""
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
import torch
from torch import Tensor

from diffusion_editor.diffusion.inversion import DDIMInverter, InversionResult
from diffusion_editor.diffusion.sampler import DDIMSampler, SamplingResult
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.editing.strength import StrengthSelection, select_edit_start
from diffusion_editor.models.noise_predictor import UNetNoisePredictor
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.text_encoder import TextEncoderWrapper
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.pipelines.conditioning import encode_conditioning
from diffusion_editor.utils.image_utils import image_to_tensor
from diffusion_editor.diffusion.refinement import validate_refinement


@dataclass(frozen=True)
class GlobalEditingResult:
    original: Tensor
    vae_reconstruction: Tensor
    source_reconstruction: Tensor
    edited_image: Tensor
    selection: StrengthSelection
    inversion: InversionResult | None
    source_sampling: SamplingResult
    target_sampling: SamplingResult


class EditingPipeline:
    def __init__(self, backend: SDBackend) -> None:
        self.backend = backend
        self.vae = VAEWrapper(backend.vae)
        self.text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
        self.predict_noise = UNetNoisePredictor(backend.unet)

    @torch.inference_mode()
    def edit_global(self, image: Image.Image | Tensor, source_prompt: str | Sequence[str],
                    target_prompt: str | Sequence[str], strength: float = 0.5,
                    guidance_scale: float = 7.5, num_steps: int = 50, *,
                    inversion_guidance_scale: float = 1.0,
                    negative_prompt: str | Sequence[str] = "",
                    return_trajectory: bool = False, inversion_refinement_steps: int = 0,
                    inversion_refinement_tolerance: float = 1e-6) -> GlobalEditingResult:
        """Invert with source, select stored level, sample source and target controls.

        Both reverse branches use the same selected latent, schedule and guidance.
        Strength 0 returns VAE-only images and skips inversion/U-Net entirely.
        This method has no spatial preservation or pixel compositing guarantee.
        Refinement is opt-in here to preserve the measured Phase 4 editing baseline.
        """
        original = image_to_tensor(image).detach().clone()
        validate_refinement(inversion_refinement_steps, inversion_refinement_tolerance)
        schedule = DDIMSchedule.from_scheduler(self.backend.scheduler, num_steps)
        selection = select_edit_start(schedule, strength)
        batch_size = original.shape[0]
        source_inversion = encode_conditioning(self.text, source_prompt, negative_prompt,
                                               inversion_guidance_scale, batch_size)
        source_reverse = (source_inversion if inversion_guidance_scale == guidance_scale
                          else encode_conditioning(self.text, source_prompt, negative_prompt,
                                                   guidance_scale, batch_size))
        target_reverse = encode_conditioning(self.text, target_prompt, negative_prompt,
                                              guidance_scale, batch_size)
        clean = self.vae.encode_image(original)
        vae_image = self.vae.decode_latents(clean)
        inversion = (DDIMInverter(schedule, self.predict_noise).invert(
                         clean, source_inversion, inversion_guidance_scale,
                         refinement_steps=inversion_refinement_steps,
                         refinement_tolerance=inversion_refinement_tolerance)
                     if selection.reverse_steps else None)
        start = clean if inversion is None else inversion.trajectory[selection.start_timestep].to(self.backend.device)
        sampler = DDIMSampler(schedule, self.predict_noise)
        source_sampling = sampler.sample(start, source_reverse, guidance_scale,
                                         start_timestep=selection.start_timestep, return_trajectory=return_trajectory)
        target_sampling = sampler.sample(start, target_reverse, guidance_scale,
                                         start_timestep=selection.start_timestep, return_trajectory=return_trajectory)
        source_image = self.vae.decode_latents(source_sampling.final_latent) if selection.reverse_steps else vae_image
        edited = self.vae.decode_latents(target_sampling.final_latent) if selection.reverse_steps else vae_image
        if source_image.shape != original.shape or edited.shape != original.shape:
            raise RuntimeError("global editing output dimensions do not match the input")
        return GlobalEditingResult(original, vae_image, source_image, edited, selection,
                                   inversion, source_sampling, target_sampling)
