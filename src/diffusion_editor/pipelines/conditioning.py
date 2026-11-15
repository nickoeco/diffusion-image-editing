"""Shared prompt preparation for reconstruction and editing orchestration."""
from collections.abc import Sequence

from torch import Tensor

from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.models.text_encoder import TextEncoderWrapper


def encode_conditioning(text: TextEncoderWrapper, prompt: str | Sequence[str],
                        negative_prompt: str | Sequence[str], scale: float,
                        batch_size: int) -> Tensor:
    """Validate prompt/image batch size; use conditional B or CFG 2B embeddings."""
    validate_guidance_scale(scale)
    embeddings = (text.encode_prompt(prompt) if scale == 1
                  else text.encode_prompt_pair(prompt, negative_prompt))
    if embeddings.shape[0] != batch_size * (1 if scale == 1 else 2):
        raise ValueError("prompt and image batch sizes must match")
    return embeddings
