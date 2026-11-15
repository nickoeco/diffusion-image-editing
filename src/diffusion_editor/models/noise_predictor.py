"""Raw conditioned U-Net execution; CFG and DDIM arithmetic stay outside."""
from typing import Any

import torch
from torch import Tensor


class UNetNoisePredictor:
    """Convert FP32 trajectory states to U-Net runtime dtype, return FP32 epsilon."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet
        self.unet.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, latent: Tensor, timestep: int, embeddings: Tensor) -> Tensor:
        if type(timestep) is not int or timestep < 0:
            raise ValueError("U-Net requires a nonnegative training timestep")
        parameter = next(self.unet.parameters())
        if latent.device != parameter.device:
            raise ValueError("latent must already be on the U-Net device")
        if embeddings.ndim != 3 or embeddings.shape[0] != latent.shape[0]:
            raise ValueError("conditioning must be [B,L,D] matching the latent batch")
        return self.unet(latent.to(dtype=parameter.dtype), timestep,
                         encoder_hidden_states=embeddings.to(device=parameter.device,
                                                             dtype=parameter.dtype)).sample.float()
