from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_editor.diffusion.schedule import CLEAN, DDIMSchedule


def test_coefficients_and_explicit_order():
    betas = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    schedule = DDIMSchedule(betas, (4, 2, 0))
    np.testing.assert_allclose(schedule.alphas_cumprod.numpy(), np.cumprod(1 - betas.numpy().astype(np.float64)), rtol=1e-6)
    assert schedule.inversion_timesteps == (CLEAN, 0, 2, 4)
    assert schedule.reverse_timesteps == (4, 2, 0, CLEAN)
    assert schedule.alpha(CLEAN).item() == 1
    assert schedule.alpha(0).item() != 1
    torch.testing.assert_close(schedule.alpha(2), schedule.alphas_cumprod[2])
    betas.zero_()
    assert schedule.betas[0] > 0  # Own the coefficient storage.
    with pytest.raises(ValueError, match="unknown"):
        schedule.alpha(1)


@pytest.mark.parametrize("betas", [torch.tensor([]), torch.ones(2, 2), torch.tensor([0.0]),
                                      torch.tensor([1.0]), torch.tensor([-0.1]), torch.tensor([float("nan")])])
def test_invalid_betas(betas):
    with pytest.raises(ValueError):
        DDIMSchedule(betas, (0,))


@pytest.mark.parametrize("levels", [(), (0, 1), (2, 2), (5,), (-1,), (1.5,), (True,)])
def test_invalid_levels(levels):
    with pytest.raises(ValueError):
        DDIMSchedule(torch.full((5,), 0.01), levels)


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon",
                                      clip_sample=False, thresholding=False, set_alpha_to_one=False)
        self.betas = torch.linspace(0.001, 0.02, 10)
        self.alphas_cumprod = (1 - self.betas).cumprod(0)
        self.timesteps = torch.tensor([9, 0])

    @classmethod
    def from_config(cls, config, **kwargs):
        instance = cls()
        instance.config = SimpleNamespace(**(vars(config) | kwargs))
        return instance

    def set_timesteps(self, num_steps, device):
        assert num_steps == 3 and device == "cpu"
        self.timesteps = torch.tensor([8, 5, 1])  # Deliberately nonuniform.


def test_uses_actual_grid_without_mutating_scheduler():
    scheduler = FakeScheduler()
    result = DDIMSchedule.from_scheduler(scheduler, 3)
    assert result.inversion_timesteps == (-1, 1, 5, 8)
    assert scheduler.timesteps.tolist() == [9, 0]
    assert scheduler.config.set_alpha_to_one is False
    torch.testing.assert_close(result.alphas_cumprod, scheduler.alphas_cumprod)


@pytest.mark.parametrize("count", [0, 11, True, 1.5])
def test_invalid_step_count(count):
    with pytest.raises(ValueError, match="num_steps"):
        DDIMSchedule.from_scheduler(FakeScheduler(), count)


@pytest.mark.parametrize("key,value", [("prediction_type", "v_prediction"), ("clip_sample", True), ("thresholding", True)])
def test_unsupported_scheduler_semantics(key, value):
    scheduler = FakeScheduler()
    setattr(scheduler.config, key, value)
    with pytest.raises(ValueError):
        DDIMSchedule.from_scheduler(scheduler, 3)
