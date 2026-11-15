"""CLIP token-level conditioning with an explicit CFG batch contract."""
from collections.abc import Sequence
from typing import Any
import warnings

import torch
from torch import Tensor


def _prompt_batch(prompt: str | Sequence[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    if not isinstance(prompt, Sequence) or not prompt or any(not isinstance(p, str) for p in prompt):
        raise ValueError("prompt must be a string or a nonempty sequence of strings")
    return list(prompt)


class TextEncoderWrapper:
    """Return token embeddings [B,L,D] on CLIP's device and dtype.

    Use last_hidden_state, not the pooled text projection. Long prompts are
    explicitly warned about and truncated to tokenizer.model_max_length.
    """

    def __init__(self, tokenizer: Any, text_encoder: Any) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_prompt(self, prompt: str | Sequence[str]) -> Tensor:
        prompts = _prompt_batch(prompt)
        length = self.tokenizer.model_max_length
        raw = self.tokenizer(prompts, padding=False, truncation=False)
        if any(len(ids) > length for ids in raw["input_ids"]):
            warnings.warn(f"Prompt truncated to {length} tokens (including special tokens)", UserWarning, stacklevel=2)
        tokens = self.tokenizer(prompts, padding="max_length", max_length=length,
                                truncation=True, return_tensors="pt")
        device = next(self.text_encoder.parameters()).device
        kwargs = {}
        if getattr(self.text_encoder.config, "use_attention_mask", False):
            kwargs["attention_mask"] = tokens["attention_mask"].to(device)
        embedding = self.text_encoder(tokens["input_ids"].to(device), **kwargs).last_hidden_state
        expected = (len(prompts), length, self.text_encoder.config.hidden_size)
        if embedding.shape != expected or not torch.isfinite(embedding).all():
            raise RuntimeError("CLIP produced invalid conditioning shape or nonfinite values")
        return embedding

    def encode_prompt_pair(
        self, prompt: str | Sequence[str], negative_prompt: str | Sequence[str] = "",
    ) -> Tensor:
        """Return [2B,L,D] ordered [all unconditional, all conditional].

        A scalar negative prompt broadcasts over the positive batch. A sequence
        must have exactly the same batch length. This does not apply CFG itself.
        """
        prompts = _prompt_batch(prompt)
        negatives = ([negative_prompt] * len(prompts) if isinstance(negative_prompt, str)
                     else _prompt_batch(negative_prompt))
        if len(negatives) != len(prompts):
            raise ValueError("negative prompt batch length must match the prompt batch")
        return self.encode_prompt(negatives + prompts)
