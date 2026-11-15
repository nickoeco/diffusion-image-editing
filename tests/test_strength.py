import math

import pytest
import torch

from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.editing.strength import select_edit_start


@pytest.mark.parametrize("strength,k,t", [(0, 0, -1), (1e-8, 1, 0), (0.25, 1, 0),
                                         (0.5, 2, 2), (0.75, 3, 7), (1, 4, 11)])
def test_discrete_strength_and_actual_noise_level(strength, k, t):
    schedule = DDIMSchedule(torch.linspace(0.01, 0.1, 12), (11, 7, 2, 0))
    selected = select_edit_start(schedule, strength)
    assert selected.reverse_steps == k
    assert selected.start_timestep == t
    assert selected.alpha_cumprod == schedule.alpha(t).item()
    assert selected.noise_std == pytest.approx(math.sqrt(1 - schedule.alpha(t).item()))
    assert len(schedule.reverse_timesteps[schedule.reverse_timesteps.index(t):]) == k + 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_strength(bad):
    with pytest.raises(ValueError, match="strength"):
        select_edit_start(DDIMSchedule(torch.full((5,), 0.01), (4, 0)), bad)


def test_single_step_and_monotonic_selection():
    one = DDIMSchedule(torch.full((5,), 0.01), (4,))
    assert select_edit_start(one, 0.01).start_timestep == 4
    schedule = DDIMSchedule(torch.full((12,), 0.01), (11, 7, 2, 0))
    selections = [select_edit_start(schedule, strength) for strength in (0, 0.1, 0.25, 0.5, 0.75, 1)]
    assert all(a.noise_std <= b.noise_std for a, b in zip(selections, selections[1:]))


@pytest.mark.parametrize("strength,steps", [(0.14, 7), (0.28, 14), (0.58, 29)])
def test_decimal_bin_boundary_avoids_float_ceil_off_by_one(strength, steps):
    schedule = DDIMSchedule(torch.full((50,), 0.01), tuple(reversed(range(50))))
    assert select_edit_start(schedule, strength).reverse_steps == steps
