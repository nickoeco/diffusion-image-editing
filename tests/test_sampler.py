import numpy as np
import pytest
import torch

from diffusion_editor.diffusion.inversion import DDIMInverter
from diffusion_editor.diffusion.sampler import DDIMSampler, ddim_sampling_step
from diffusion_editor.diffusion.schedule import DDIMSchedule


@pytest.mark.parametrize("alpha_t,alpha_s", [(0.02, 0.7), (0.7, 0.99), (0.99, 1.0)])
def test_reverse_equations_float64(alpha_t, alpha_s):
    z = torch.linspace(-1, 1, 24).reshape(1, 3, 2, 4)
    eps = torch.full_like(z, 0.2)
    actual = ddim_sampling_step(z, eps, torch.tensor(alpha_t), torch.tensor(alpha_s))
    x0 = (z.numpy().astype(np.float64) - np.sqrt(1 - alpha_t) * eps.numpy()) / np.sqrt(alpha_t)
    expected = np.sqrt(alpha_s) * x0 + np.sqrt(1 - alpha_s) * eps.numpy()
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7, rtol=2e-6)


def test_constant_noise_roundtrip_and_timestep_queries():
    schedule = DDIMSchedule(torch.linspace(0.01, 0.1, 12), (11, 7, 2, 0))
    original = torch.linspace(-1, 1, 32).reshape(2, 4, 2, 2)
    calls = []
    def predictor(z, t, e):
        calls.append(t)
        return torch.full_like(z, 0.2)
    embeddings = torch.zeros(2, 3, 5)
    inversion = DDIMInverter(schedule, predictor).invert(original, embeddings)
    calls.clear()
    result = DDIMSampler(schedule, predictor).sample(inversion.final_latent, embeddings, return_trajectory=True)
    assert calls == [11, 7, 2, 0]
    assert result.timesteps == (11, 7, 2, 0, -1)
    assert tuple(result.trajectory) == result.timesteps
    torch.testing.assert_close(result.final_latent, original)
    torch.testing.assert_close(result.trajectory[-1], original)
    repeat = DDIMSampler(schedule, predictor).sample(inversion.final_latent, embeddings)
    assert repeat.trajectory is None
    torch.testing.assert_close(repeat.final_latent, result.final_latent, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0, 1, 5])
def test_cfg_inversion_sampling_analytic_roundtrip(scale):
    schedule = DDIMSchedule(torch.full((10,), 0.02), (9, 5, 1))
    clean = torch.zeros(1, 4, 2, 2)
    pair = torch.tensor([0.1, 0.4]).reshape(2, 1, 1)
    def predict(z, t, e):
        return torch.zeros_like(z) + e[:, 0, 0].reshape(-1, 1, 1, 1)
    inv = DDIMInverter(schedule, predict).invert(clean, pair, scale)
    expected_noise = 0.1 + scale * 0.3
    alpha = schedule.alpha(9)
    torch.testing.assert_close(inv.final_latent, torch.full_like(clean, (1 - alpha).sqrt() * expected_noise))
    sampled = DDIMSampler(schedule, predict).sample(inv.final_latent, pair, scale)
    torch.testing.assert_close(sampled.final_latent, clean, atol=1e-6, rtol=0)


@pytest.mark.parametrize("coefficients", [(1., 0.8), (0., 1.), (0.8, 0.8), (0.8, float("nan"))])
def test_invalid_reverse_coefficients(coefficients):
    z = torch.zeros(1, 4, 2, 2)
    with pytest.raises(ValueError, match="scalar coefficients"):
        ddim_sampling_step(z, z, *map(torch.tensor, coefficients))


def test_sampler_rejects_bad_latents():
    sampler = DDIMSampler(DDIMSchedule(torch.full((5,), 0.01), (4, 0)), lambda z, t, e: z)
    for z in (torch.zeros(4, 2, 2), torch.zeros(1, 4, 2, 2).long(), torch.full((1, 4, 2, 2), float("inf"))):
        with pytest.raises(ValueError):
            sampler.sample(z, torch.ones(1, 2, 3))
