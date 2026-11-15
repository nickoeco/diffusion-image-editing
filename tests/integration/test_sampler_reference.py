import pytest


@pytest.mark.integration
@pytest.mark.parametrize("steps", [1, 20, 50, 100])
def test_sampling_against_diffusers(steps):
    import torch
    from diffusers import DDIMScheduler
    from diffusion_editor.diffusion.sampler import ddim_sampling_step
    from diffusion_editor.diffusion.schedule import DDIMSchedule

    reference = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, steps_offset=1, set_alpha_to_one=True)
    reference.set_timesteps(steps)
    schedule = DDIMSchedule.from_scheduler(reference, steps)
    z = torch.linspace(-1, 1, 32).reshape(1, 4, 2, 4)
    eps = torch.full_like(z, 0.2)
    for t, s in zip(schedule.reverse_timesteps, schedule.reverse_timesteps[1:]):
        actual = ddim_sampling_step(z, eps, schedule.alpha(t), schedule.alpha(s))
        expected = reference.step(eps, t, z, eta=0).prev_sample
        torch.testing.assert_close(actual, expected)
        z = actual
