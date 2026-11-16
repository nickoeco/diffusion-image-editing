import pytest
import torch

from diffusion_editor.diffusion.sampler import DDIMSampler
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.editing.blending import SourceLatentBlender, blend_latents, composite_pixels


@pytest.mark.parametrize("weight", [0., 1., 0.25])
def test_zero_full_and_soft_blend(weight):
    edited, source = torch.ones(2, 4, 3, 5) * 3, torch.ones(2, 4, 3, 5) * -2
    mask = torch.full((2, 1, 3, 5), weight)
    result = blend_latents(edited.half(), source, mask)
    torch.testing.assert_close(result, torch.full_like(source, weight * 3 + (1 - weight) * -2))
    assert result.dtype == torch.float32


def test_partial_blend_and_pixel_preservation():
    original, edited = torch.rand(1, 3, 8, 12), torch.rand(1, 3, 8, 12)
    mask = torch.zeros(1, 1, 8, 12)
    mask[:, :, 2:6, 4:8] = 1
    output = composite_pixels(edited, original, mask)
    torch.testing.assert_close(output[:, :, 2:6, 4:8], edited[:, :, 2:6, 4:8], rtol=0, atol=0)
    background = (mask == 0).expand_as(original)
    torch.testing.assert_close(output[background], original[background], rtol=0, atol=0)


def test_sampler_blends_at_updated_ids_including_clean():
    schedule = DDIMSchedule(torch.full((5,), 0.02), (4, 2, 0))
    initial = torch.ones(1, 4, 2, 3) * 7
    source = {2: torch.ones_like(initial) * 2, 0: torch.zeros_like(initial), -1: torch.ones_like(initial) * -0.5}
    blender = SourceLatentBlender(source, torch.zeros(1, 1, 2, 3), (2, 0, -1), initial)
    calls = []
    def after_step(updated, timestep):
        calls.append(timestep)
        return blender(updated, timestep)
    result = DDIMSampler(schedule, lambda z, t, e: torch.zeros_like(z)).sample(
        initial, torch.ones(1, 2, 3), return_trajectory=True, after_step=after_step)
    assert calls == [2, 0, -1]
    for t in calls:
        torch.testing.assert_close(result.trajectory[t], source[t], rtol=0, atol=0)
    torch.testing.assert_close(result.final_latent, source[-1], rtol=0, atol=0)


def test_missing_terminal_source_and_invalid_shapes_fail():
    z = torch.ones(1, 4, 2, 3)
    with pytest.raises(ValueError, match="missing updated timestep -1"):
        SourceLatentBlender({0: z}, torch.ones(1, 1, 2, 3), (0, -1), z)
    with pytest.raises(ValueError, match="shape"):
        blend_latents(z, z, torch.ones(1, 4, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        blend_latents(z, z * float("nan"), torch.ones(1, 1, 2, 3))


@pytest.mark.parametrize("callback", [lambda z, t: z.half(), lambda z, t: z[:, :1],
                                          lambda z, t: z + float("nan")])
def test_invalid_sampler_hook_results(callback):
    schedule = DDIMSchedule(torch.full((5,), 0.02), (4, 0))
    with pytest.raises(ValueError, match="after_step"):
        DDIMSampler(schedule, lambda z, t, e: z * 0).sample(
            torch.ones(1, 4, 2, 3), torch.ones(1, 2, 3), after_step=callback)
