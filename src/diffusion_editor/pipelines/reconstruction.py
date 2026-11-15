"""Same-prompt image reconstruction with independent VAE and DDIM outputs."""
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Mapping

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
from diffusion_editor.diffusion.refinement import RefinementDiagnostics, validate_refinement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionFidelityCheck:
    selected_method: str
    refined_latent_mse: float
    plain_latent_mse: float
    candidate_refinement: Mapping[int, RefinementDiagnostics]


@dataclass(frozen=True)
class ReconstructionResult:
    original: Tensor
    vae_reconstruction: Tensor
    ddim_reconstruction: Tensor
    inversion: InversionResult
    sampling: SamplingResult
    fidelity_check: ReconstructionFidelityCheck | None = None


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
                    return_trajectory: bool = False, inversion_refinement_steps: int = 12,
                    inversion_refinement_tolerance: float = 1e-6) -> ReconstructionResult:
        """No editing: source prompt is used for BOTH inversion and sampling.

        Guidance 1 uses conditional B embeddings; other scales use a 2B CFG batch.
        Keeping the two guidance settings explicit avoids hiding reconstruction
        changes caused by guidance rather than the VAE or time grid.
        Residual refinement defaults to 12 evaluations per step, stopping early
        at the stated RMS tolerance; set 0 for the original approximate DDIM.
        Refined reconstruction also runs the plain path and chooses the lower
        terminal roundtrip latent MSE; local residual improvement is insufficient.
        """
        validate_guidance_scale(guidance_scale)
        validate_guidance_scale(inversion_guidance_scale)
        validate_refinement(inversion_refinement_steps, inversion_refinement_tolerance)
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
            latent, inversion_embeddings, inversion_guidance_scale,
            refinement_steps=inversion_refinement_steps,
            refinement_tolerance=inversion_refinement_tolerance)
        sampling = DDIMSampler(schedule, self.predict_noise).sample(
            inversion.final_latent.to(self.backend.device),
            sampling_embeddings, guidance_scale,
            return_trajectory=return_trajectory)
        fidelity = None
        if inversion_refinement_steps:
            plain_inversion = DDIMInverter(schedule, self.predict_noise).invert(
                latent, inversion_embeddings, inversion_guidance_scale)
            plain_sampling = DDIMSampler(schedule, self.predict_noise).sample(
                plain_inversion.final_latent.to(self.backend.device), sampling_embeddings, guidance_scale,
                return_trajectory=return_trajectory)
            refined_mse = float((sampling.final_latent - latent).square().mean())
            plain_mse = float((plain_sampling.final_latent - latent).square().mean())
            method = "residual_refined_ddim" if refined_mse <= plain_mse else "plain_ddim_fallback"
            fidelity = ReconstructionFidelityCheck(method, refined_mse, plain_mse, inversion.refinement)
            if method == "plain_ddim_fallback":
                logger.warning("Refined roundtrip MSE %.6g exceeds plain %.6g; selecting plain DDIM reconstruction",
                               refined_mse, plain_mse)
                inversion, sampling = plain_inversion, plain_sampling
        ddim_image = self.vae.decode_latents(sampling.final_latent)
        if vae_image.shape != original.shape or ddim_image.shape != original.shape:
            raise RuntimeError("reconstruction output dimensions do not match the input")
        return ReconstructionResult(original, vae_image, ddim_image, inversion, sampling, fidelity)
