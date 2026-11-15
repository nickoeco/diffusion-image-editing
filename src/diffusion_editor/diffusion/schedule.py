"""Training coefficients and explicit inference levels shared by future samplers."""
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

CLEAN = -1


@dataclass(frozen=True)
class DDIMSchedule:
    """CPU FP32 coefficients, descending training timestep IDs, clean boundary -1.

    No adjacent-integer assumption is made. The terminal inverted state is the
    highest selected timestep, not necessarily training timestep T-1 or pure noise.
    Construct from the pretrained scheduler in production; explicit betas support
    small offline mathematical tests. Treat stored coefficient tensors as read-only.
    """
    betas: Tensor
    sampling_timesteps: tuple[int, ...]
    alphas: Tensor = field(init=False, repr=False)
    alphas_cumprod: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.betas, Tensor) or self.betas.ndim != 1 or self.betas.numel() == 0:
            raise ValueError("betas must be a nonempty 1D tensor")
        betas = self.betas.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(betas).all() or (betas <= 0).any() or (betas >= 1).any():
            raise ValueError("epsilon DDIM requires finite betas strictly between 0 and 1")
        levels = tuple(self.sampling_timesteps)
        if not levels or any(type(t) is not int or t < 0 or t >= len(betas) for t in levels):
            raise ValueError("timesteps must be nonempty integer training IDs within beta range")
        if any(t <= s for t, s in zip(levels, levels[1:])):
            raise ValueError("sampling timesteps must be strictly descending and unique")
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        if (cumulative <= 0).any() or (cumulative >= 1).any():
            raise ValueError("alpha products must remain strictly between 0 and 1 in FP32")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "sampling_timesteps", levels)
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "alphas_cumprod", cumulative)

    @classmethod
    def from_scheduler(cls, scheduler: Any, num_steps: int) -> "DDIMSchedule":
        """Clone scheduler config; preserve its actual inference grid and betas.

        Only the clean endpoint convention is overridden (set_alpha_to_one=True).
        The original scheduler is not mutated. No Diffusers stepping is executed.
        """
        if type(num_steps) is not int or not 1 <= num_steps <= scheduler.config.num_train_timesteps:
            raise ValueError("num_steps must be an integer in [1,num_train_timesteps]")
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("Only epsilon-prediction schedules are supported")
        if getattr(scheduler.config, "clip_sample", False) or getattr(scheduler.config, "thresholding", False):
            raise ValueError("Unclipped DDIM equations require clip_sample=False and thresholding=False")
        reference = type(scheduler).from_config(scheduler.config, set_alpha_to_one=True)
        reference.set_timesteps(num_steps, device="cpu")
        result = cls(reference.betas, tuple(int(t) for t in reference.timesteps.tolist()))
        if len(result.sampling_timesteps) != num_steps:
            raise ValueError("scheduler returned an unexpected timestep count")
        if not torch.equal(result.alphas_cumprod, reference.alphas_cumprod.cpu().float()):
            raise ValueError("scheduler alpha products do not match its training betas")
        return result

    @property
    def inversion_timesteps(self) -> tuple[int, ...]:
        return (CLEAN, *reversed(self.sampling_timesteps))

    @property
    def reverse_timesteps(self) -> tuple[int, ...]:
        return (*self.sampling_timesteps, CLEAN)

    def alpha(self, timestep: int, *, device: torch.device | str = "cpu") -> Tensor:
        """Coefficient for an explicit selected level; CLEAN has alpha exactly 1."""
        if type(timestep) is not int or timestep not in self.inversion_timesteps:
            raise ValueError(f"unknown trajectory timestep: {timestep}")
        value = torch.tensor(1.0) if timestep == CLEAN else self.alphas_cumprod[timestep]
        return value.to(device=device, dtype=torch.float32)
