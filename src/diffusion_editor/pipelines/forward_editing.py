"""Forward-noising Img2Img, four-channel masked inpainting and canvas outpainting."""
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
import torch
from torch import Tensor

from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.diffusion.noising import forward_noise
from diffusion_editor.diffusion.sampler import DDIMSampler, SamplingResult
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.editing.blending import blend_latents, composite_pixels
from diffusion_editor.editing.canvas import OutpaintCanvas, prepare_outpaint_canvas
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.editing.strength import StrengthSelection, select_forward_start
from diffusion_editor.models.noise_predictor import UNetNoisePredictor
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.text_encoder import TextEncoderWrapper
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.pipelines.conditioning import encode_conditioning
from diffusion_editor.utils.image_utils import image_to_tensor


@dataclass(frozen=True)
class ForwardEditingResult:
    mode: str
    original: Tensor
    vae_reconstruction: Tensor
    edited_image: Tensor
    uncomposited_image: Tensor
    selection: StrengthSelection
    sampling: SamplingResult
    noise: Tensor
    pixel_mask: Tensor | None
    latent_mask: Tensor | None
    canvas: OutpaintCanvas | None = None


class ForwardEditingPipeline:
    """No inversion or source text. Freeze components; use explicit eta=0 DDIM/CFG.

    One CPU-generated FP32 Gaussian tensor seeds q(z_t|z_0) and each known-region
    marginal at the UPDATED ID. Only z0 and epsilon are retained for blending.
    """

    def __init__(self, backend: SDBackend) -> None:
        self.backend = backend
        self.vae = VAEWrapper(backend.vae)
        self.text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
        self.predict_noise = UNetNoisePredictor(backend.unet)

    def img2img(self, image: Image.Image | Tensor, prompt: str | Sequence[str], *, strength: float = 0.75,
                guidance_scale: float = 7.5, num_steps: int = 50, seed: int = 42,
                negative_prompt: str | Sequence[str] = "", return_trajectory: bool = False) -> ForwardEditingResult:
        return self._run(image, prompt, strength=strength, guidance_scale=guidance_scale, num_steps=num_steps,
                         seed=seed, negative_prompt=negative_prompt, return_trajectory=return_trajectory)

    def inpaint(self, image: Image.Image | Tensor, mask: Image.Image | Tensor, prompt: str | Sequence[str], *,
                strength: float = 0.85, guidance_scale: float = 7.5, num_steps: int = 50, seed: int = 42,
                negative_prompt: str | Sequence[str] = "", binary_mask: bool = True, feather_radius: float = 0,
                latent_blending: bool = True, preserve_background: bool = True,
                return_trajectory: bool = False) -> ForwardEditingResult:
        """Base-model inpainting, not a nine-channel inpainting checkpoint."""
        return self._run(image, prompt, mask=mask, strength=strength, guidance_scale=guidance_scale,
                         num_steps=num_steps, seed=seed, negative_prompt=negative_prompt,
                         binary_mask=binary_mask, feather_radius=feather_radius, latent_blending=latent_blending,
                         preserve_background=preserve_background, return_trajectory=return_trajectory)

    def outpaint(self, image: Image.Image | Tensor, prompt: str | Sequence[str], *, left: int = 0,
                 right: int = 128, top: int = 0, bottom: int = 0, overlap: int = 0, fill: str = "edge",
                 strength: float = 1.0, guidance_scale: float = 7.5, num_steps: int = 50, seed: int = 42,
                 negative_prompt: str | Sequence[str] = "", feather_radius: float = 0,
                 latent_blending: bool = True, preserve_background: bool = True,
                 return_trajectory: bool = False) -> ForwardEditingResult:
        canvas = prepare_outpaint_canvas(image, left=left, right=right, top=top, bottom=bottom, overlap=overlap, fill=fill)
        return self._run(canvas.pixels, prompt, mask=canvas.mask, strength=strength, guidance_scale=guidance_scale,
                         num_steps=num_steps, seed=seed, negative_prompt=negative_prompt, binary_mask=False,
                         feather_radius=feather_radius, latent_blending=latent_blending,
                         preserve_background=preserve_background, return_trajectory=return_trajectory, canvas=canvas)

    @torch.inference_mode()
    def _run(self, image: Image.Image | Tensor, prompt: str | Sequence[str], *, strength: float,
             guidance_scale: float, num_steps: int, seed: int, negative_prompt: str | Sequence[str],
             return_trajectory: bool, mask: Image.Image | Tensor | None = None,
             binary_mask: bool = True, feather_radius: float = 0, latent_blending: bool = True,
             preserve_background: bool = True, canvas: OutpaintCanvas | None = None) -> ForwardEditingResult:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32)")
        if any(type(flag) is not bool for flag in (return_trajectory, binary_mask, latent_blending, preserve_background)):
            raise ValueError("Mask, compositing and trajectory switches must be boolean")
        validate_guidance_scale(guidance_scale)
        original = image_to_tensor(image).detach().clone()
        schedule = DDIMSchedule.from_scheduler(self.backend.scheduler, num_steps)
        selection = select_forward_start(schedule, strength)
        # Validate mask and conditioning before diffusion work. Mask latent size is
        # derived from model config, never from an assumed SD spatial factor.
        height, width = original.shape[-2:]
        factor = self.vae.spatial_factor
        if height % factor or width % factor:
            raise ValueError(f"Image/canvas dimensions must be multiples of {factor}; resize explicitly")
        prepared = prepare_mask(mask, (height, width), (height // factor, width // factor), original.shape[0],
                                binary=binary_mask, feather_radius=feather_radius) if mask is not None else None
        embeddings = encode_conditioning(self.text, prompt, negative_prompt, guidance_scale, original.shape[0])
        clean = self.vae.encode_image(original)
        vae_image = self.vae.decode_latents(clean)
        noise = torch.randn(clean.shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
        epsilon = noise.to(clean.device)
        start = forward_noise(clean, epsilon, schedule.alpha(selection.start_timestep, device=clean.device))
        after_step = None
        if prepared is not None and latent_blending:
            latent_mask = prepared.latent.to(clean.device)
            def after_step(updated: Tensor, timestep: int) -> Tensor:
                known = forward_noise(clean, epsilon, schedule.alpha(timestep, device=clean.device))
                return blend_latents(updated, known, latent_mask)
        sampling = DDIMSampler(schedule, self.predict_noise).sample(start, embeddings, guidance_scale,
            start_timestep=selection.start_timestep, after_step=after_step, return_trajectory=return_trajectory)
        raw = self.vae.decode_latents(sampling.final_latent) if selection.reverse_steps else vae_image
        output = composite_pixels(raw, original, prepared.pixel) if prepared is not None and preserve_background else raw
        mode = "outpaint" if canvas is not None else "inpaint" if prepared is not None else "img2img"
        return ForwardEditingResult(mode, original, vae_image, output, raw, selection, sampling, noise,
                                    prepared.pixel if prepared is not None else None,
                                    prepared.latent if prepared is not None else None, canvas)
