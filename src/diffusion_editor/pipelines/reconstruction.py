"""Same-prompt image reconstruction with independent VAE and DDIM outputs."""
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
import torch
from torch import Tensor

from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.diffusion.inversion import DDIMInverter, InversionResult
from diffusion_editor.diffusion.sampler import DDIMSampler, SamplingResult
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.models.noise_predictor import UNetNoisePredictor
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.text_encoder import TextEncoderWrapper
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.utils.image_utils import image_to_tensor
from diffusion_editor.pipelines.conditioning import encode_conditioning


@dataclass(frozen=True)
class ReconstructionResult:
    original: Tensor
    vae_reconstruction: Tensor
    ddim_reconstruction: Tensor
    inversion: InversionResult
    sampling: SamplingResult


class ReconstructionPipeline:
    def __init__(self, backend: SDBackend) -> None:
        self.backend = backend
        self.vae = VAEWrapper(backend.vae)
        self.text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
        self.predict_noise = UNetNoisePredictor(backend.unet)

    @torch.inference_mode()
    def reconstruct(self, image: Image.Image | Tensor, source_prompt: str | Sequence[str],
                    num_steps: int = 50, guidance_scale: float = 1.0,
                    *, inversion_guidance_scale: float = 1.0,
                    negative_prompt: str | Sequence[str] = "",
                    return_trajectory: bool = False) -> ReconstructionResult:
        """No editing: source prompt is used for BOTH inversion and sampling.

        Guidance 1 uses conditional B embeddings; other scales use a 2B CFG batch.
        Keeping the two guidance settings explicit avoids hiding reconstruction
        changes caused by guidance rather than the VAE or time grid.
        """
        validate_guidance_scale(guidance_scale)
        validate_guidance_scale(inversion_guidance_scale)
        original = image_to_tensor(image).detach().clone()
        schedule = DDIMSchedule.from_scheduler(self.backend.scheduler, num_steps)
        inversion_embeddings = encode_conditioning(self.text, source_prompt, negative_prompt,
                                                    inversion_guidance_scale, original.shape[0])
        sampling_embeddings = (inversion_embeddings if inversion_guidance_scale == guidance_scale
                               else encode_conditioning(self.text, source_prompt, negative_prompt,
                                                        guidance_scale, original.shape[0]))
        latent = self.vae.encode_image(original)
        vae_image = self.vae.decode_latents(latent)
        inversion = DDIMInverter(schedule, self.predict_noise).invert(
            latent, inversion_embeddings, inversion_guidance_scale)
        sampling = DDIMSampler(schedule, self.predict_noise).sample(
            inversion.final_latent.to(self.backend.device),
            sampling_embeddings, guidance_scale,
            return_trajectory=return_trajectory)
        ddim_image = self.vae.decode_latents(sampling.final_latent)
        if vae_image.shape != original.shape or ddim_image.shape != original.shape:
            raise RuntimeError("reconstruction output dimensions do not match the input")
        return ReconstructionResult(original, vae_image, ddim_image, inversion, sampling)
