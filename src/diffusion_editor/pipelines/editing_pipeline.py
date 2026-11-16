"""Global source/target text editing using stored DDIM inversion levels."""
from collections.abc import Sequence
from collections.abc import Callable
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
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.editing.blending import SourceLatentBlender, composite_pixels


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


@dataclass(frozen=True)
class LocalEditingResult(GlobalEditingResult):
    pixel_mask: Tensor
    latent_mask: Tensor
    uncomposited_image: Tensor
    uncomposited_source: Tensor
    latent_blending: bool
    preserve_background: bool


@dataclass(frozen=True)
class _PreparedEdit:
    original: Tensor
    vae_image: Tensor
    clean: Tensor
    schedule: DDIMSchedule
    selection: StrengthSelection
    inversion: InversionResult | None
    start: Tensor
    source_reverse: Tensor
    target_reverse: Tensor
    guidance_scale: float


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
        prepared = self._prepare_edit(image, source_prompt, target_prompt, strength, guidance_scale, num_steps,
                                      inversion_guidance_scale, negative_prompt,
                                      inversion_refinement_steps, inversion_refinement_tolerance)
        return self._sample_edit(prepared, return_trajectory)

    def _prepare_edit(self, image: Image.Image | Tensor, source_prompt: str | Sequence[str],
                      target_prompt: str | Sequence[str], strength: float, guidance_scale: float, num_steps: int,
                      inversion_guidance_scale: float, negative_prompt: str | Sequence[str],
                      inversion_refinement_steps: int, inversion_refinement_tolerance: float) -> _PreparedEdit:
        """Share one source inversion/conditioning path between global and local edits."""
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
        return _PreparedEdit(original, vae_image, clean, schedule, selection, inversion, start,
                             source_reverse, target_reverse, guidance_scale)

    def _sample_edit(self, prepared: _PreparedEdit, return_trajectory: bool,
                     after_step: Callable[[Tensor, int], Tensor] | None = None) -> GlobalEditingResult:
        p = prepared
        sampler = DDIMSampler(p.schedule, self.predict_noise)
        source_sampling = sampler.sample(p.start, p.source_reverse, p.guidance_scale,
                                         start_timestep=p.selection.start_timestep, return_trajectory=return_trajectory,
                                         after_step=after_step)
        target_sampling = sampler.sample(p.start, p.target_reverse, p.guidance_scale,
                                         start_timestep=p.selection.start_timestep, return_trajectory=return_trajectory,
                                         after_step=after_step)
        source_image = self.vae.decode_latents(source_sampling.final_latent) if p.selection.reverse_steps else p.vae_image
        edited = self.vae.decode_latents(target_sampling.final_latent) if p.selection.reverse_steps else p.vae_image
        if source_image.shape != p.original.shape or edited.shape != p.original.shape:
            raise RuntimeError("global editing output dimensions do not match the input")
        return GlobalEditingResult(p.original, p.vae_image, source_image, edited, p.selection,
                                   p.inversion, source_sampling, target_sampling)

    @torch.inference_mode()
    def edit_local(self, image: Image.Image | Tensor, source_prompt: str | Sequence[str],
                   target_prompt: str | Sequence[str], mask: Image.Image | Tensor,
                   strength: float = 0.5, guidance_scale: float = 7.5, num_steps: int = 50, *,
                   binary_mask: bool = True, feather_radius: float = 0,
                   latent_blending: bool = True, preserve_background: bool = True,
                   inversion_guidance_scale: float = 1.0, negative_prompt: str | Sequence[str] = "",
                   return_trajectory: bool = False, inversion_refinement_steps: int = 0,
                   inversion_refinement_tolerance: float = 1e-6) -> LocalEditingResult:
        """Blend after each update against source at the UPDATED ID, then optionally composite.

        Mask 1 edits and 0 preserves. Pixel compositing uses the effective feathered
        pixel mask; outside its zero support preservation is exact. Latent-only
        preservation remains subject to VAE spatial coupling and reconstruction loss.
        """
        if type(latent_blending) is not bool or type(preserve_background) is not bool:
            raise ValueError("latent_blending and preserve_background must be boolean")
        original = image_to_tensor(image)
        height, width = original.shape[-2:]
        factor = self.vae.spatial_factor
        if height % factor or width % factor:
            raise ValueError(f"image dimensions must be multiples of VAE spatial factor {factor}")
        masks = prepare_mask(mask, (height, width), (height // factor, width // factor), original.shape[0],
                             binary=binary_mask, feather_radius=feather_radius)
        p = self._prepare_edit(original, source_prompt, target_prompt, strength, guidance_scale, num_steps,
                               inversion_guidance_scale, negative_prompt,
                               inversion_refinement_steps, inversion_refinement_tolerance)
        levels = p.schedule.reverse_timesteps
        updated_ids = levels[levels.index(p.selection.start_timestep) + 1:]
        source_states = p.inversion.trajectory if p.inversion else {-1: p.clean}
        blender = SourceLatentBlender(source_states, masks.latent, updated_ids, p.start) if latent_blending else None
        base = self._sample_edit(p, return_trajectory, blender)
        edited = composite_pixels(base.edited_image, p.original, masks.pixel) if preserve_background else base.edited_image
        source = composite_pixels(base.source_reconstruction, p.original, masks.pixel) if preserve_background else base.source_reconstruction
        return LocalEditingResult(base.original, base.vae_reconstruction, source, edited, base.selection,
                                  base.inversion, base.source_sampling, base.target_sampling,
                                  masks.pixel, masks.latent, base.edited_image, base.source_reconstruction,
                                  latent_blending, preserve_background)
