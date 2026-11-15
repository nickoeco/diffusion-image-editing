import pytest


@pytest.mark.integration
@pytest.mark.parametrize("steps", [1, 20, 50, 100])
def test_diffusers_inverse_step_reference(steps):
    import torch
    from diffusers import DDIMScheduler, DDIMInverseScheduler
    from diffusion_editor.diffusion.schedule import DDIMSchedule
    from diffusion_editor.diffusion.inversion import ddim_inversion_step

    # No model download: configuration-only reference for all selected step pairs.
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, steps_offset=1, set_alpha_to_one=False)
    schedule = DDIMSchedule.from_scheduler(scheduler, steps)
    reference = DDIMInverseScheduler.from_config(scheduler.config, set_alpha_to_one=True)
    reference.set_timesteps(steps)
    assert tuple(reference.timesteps.tolist()) == schedule.inversion_timesteps[1:]
    z = torch.linspace(-1, 1, 32).reshape(1, 4, 2, 4)
    eps = torch.full_like(z, 0.13)
    for t, s in zip(schedule.inversion_timesteps, schedule.inversion_timesteps[1:]):
        actual = ddim_inversion_step(z, eps, schedule.alpha(t), schedule.alpha(s))
        expected = reference.step(eps, s, z).prev_sample
        torch.testing.assert_close(actual, expected)
        z = actual


@pytest.mark.integration
@pytest.mark.parametrize("spacing", ["leading", "linspace", "trailing"])
def test_actual_scheduler_spacing_is_preserved(spacing):
    import torch
    from diffusers import DDIMScheduler
    from diffusion_editor.diffusion.schedule import DDIMSchedule

    scheduler = DDIMScheduler(clip_sample=False, timestep_spacing=spacing, steps_offset=1)
    scheduler.set_timesteps(37)
    original = scheduler.timesteps.clone()
    schedule = DDIMSchedule.from_scheduler(scheduler, 37)
    assert schedule.sampling_timesteps == tuple(original.tolist())
    assert torch.equal(scheduler.timesteps, original)


@pytest.mark.integration
def test_leading_offset_out_of_range_rejected():
    from diffusers import DDIMScheduler
    from diffusion_editor.diffusion.schedule import DDIMSchedule

    scheduler = DDIMScheduler(clip_sample=False, steps_offset=1)
    with pytest.raises(ValueError, match="within beta range"):
        DDIMSchedule.from_scheduler(scheduler, 1000)
