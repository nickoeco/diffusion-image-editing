"""Load individual SD 1.x components, never an editing pipeline."""
from dataclasses import dataclass
import logging
from typing import Any

from diffusion_editor.config import ProjectConfig

logger = logging.getLogger(__name__)


def resolve_runtime(config: ProjectConfig, torch: Any) -> tuple[str, Any]:
    """CUDA auto -> FP16, CPU auto -> FP32; reject unavailable devices."""
    cuda = torch.cuda.is_available()
    device = ("cuda" if cuda else "cpu") if config.device == "auto" else config.device
    if device == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but unavailable; check PyTorch build/driver or use cpu")
    precision = ("fp16" if device == "cuda" else "fp32") if config.precision == "auto" else config.precision
    if device == "cpu" and precision == "fp16":
        raise ValueError("CPU inference requires fp32")
    return device, torch.float16 if precision == "fp16" else torch.float32


@dataclass
class SDBackend:
    """Frozen inference components; tokenizer/scheduler stay on the host.

    VAE uses FP32 for stable encode/decode; U-Net/CLIP use runtime dtype.
    Callers must explicitly cast VAE inputs in Phase 1 wrappers.
    """
    vae: Any
    tokenizer: Any
    text_encoder: Any
    unet: Any
    scheduler: Any
    device: str
    dtype: Any
    config: ProjectConfig

    @classmethod
    def load(cls, config: ProjectConfig) -> "SDBackend":
        """Load safetensors from a pinned revision, offline by default.

        Download/cache errors propagate with their original diagnostics. Loading
        scheduler configuration does not implement or execute our DDIM algorithm.
        """
        try:
            import torch
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError("Model dependencies missing; install .[model] as documented in README") from exc

        device, dtype = resolve_runtime(config, torch)
        common = dict(revision=config.revision, cache_dir=config.cache_dir,
                      local_files_only=config.local_files_only)
        weights = dict(common, use_safetensors=True)
        # Select FP16 weight files only for the components executed in FP16.
        neural_weights = dict(weights, **({"variant": "fp16"} if dtype == torch.float16 else {}))
        logger.info("Loading %s@%s on %s (%s); offline=%s", config.model_id,
                    config.revision, device, dtype, config.local_files_only)
        scheduler = DDIMScheduler.from_pretrained(config.model_id, subfolder="scheduler", **common)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("Only epsilon-prediction SD models are supported by the planned equations")
        tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer", **common)
        vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae",
                                           torch_dtype=torch.float32, **weights)
        text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder",
                                                     torch_dtype=dtype, **neural_weights)
        unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet",
                                                  torch_dtype=dtype, **neural_weights)
        for model in (vae, text_encoder, unet):
            model.requires_grad_(False)
            model.eval()
            model.to(device)
        if unet.config.in_channels != vae.config.latent_channels:
            raise ValueError("U-Net channels do not match VAE latent channels (expected SD base model)")
        return cls(vae, tokenizer, text_encoder, unet, scheduler, device, dtype, config)
