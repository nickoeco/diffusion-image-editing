from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from diffusion_editor.diffusion.inversion import DDIMInverter, ddim_inversion_step
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.models.noise_predictor import UNetNoisePredictor


@pytest.mark.parametrize("alpha_t,alpha_s", [(1.0, 0.99), (0.99, 0.72), (0.72, 0.03)])
def test_step_against_float64_equations(alpha_t, alpha_s):
    z = torch.linspace(-1, 1, 24).reshape(1, 3, 2, 4)
    eps = torch.linspace(0.5, -0.5, 24).reshape_as(z)
    actual = ddim_inversion_step(z, eps, torch.tensor(alpha_t), torch.tensor(alpha_s))
    # Independent double-precision equation oracle, including the clean boundary.
    x0 = (z.numpy().astype(np.float64) - np.sqrt(1 - alpha_t) * eps.numpy()) / np.sqrt(alpha_t)
    expected = np.sqrt(alpha_s) * x0 + np.sqrt(1 - alpha_s) * eps.numpy()
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7, rtol=2e-6)


def test_analytic_constant_epsilon_trajectory_and_no_aliasing():
    schedule = DDIMSchedule(torch.linspace(0.001, 0.1, 12), (11, 7, 2, 0))
    clean = torch.linspace(-1, 1, 32).reshape(2, 4, 2, 2)
    eps = torch.full_like(clean, 0.3)
    calls = []
    def predict(z, t, embeddings):
        calls.append(t)
        assert embeddings.shape == (2, 3, 5)
        return eps
    inverter = DDIMInverter(schedule, predict)
    result = inverter.invert(clean, torch.ones(2, 3, 5))
    assert result.timesteps == (-1, 0, 2, 7, 11)
    assert result.prediction_timesteps == (0, 0, 2, 7)
    assert calls == [0, 0, 2, 7]
    assert len(result.trajectory) == 5
    for t in result.timesteps:
        a = schedule.alpha(t)
        expected = a.sqrt() * clean + (1 - a).sqrt() * eps
        torch.testing.assert_close(result.trajectory[t], expected)
        assert not result.trajectory[t].requires_grad
    repeated = inverter.invert(clean, torch.ones(2, 3, 5))
    torch.testing.assert_close(result.final_latent, repeated.final_latent, rtol=0, atol=0)
    snapshot = result.initial_latent.clone()
    clean.zero_()
    torch.testing.assert_close(result.initial_latent, snapshot)
    with pytest.raises(TypeError):
        result.trajectory[1] = clean
    with pytest.raises(KeyError):
        _ = result.trajectory[1]


@pytest.mark.parametrize("pair", [(1.0, 1.0), (0.3, 0.5), (1.0, 0.0), (1.1, 0.5), (float("nan"), 0.5)])
def test_invalid_step_coefficients(pair):
    z = torch.ones(1, 4, 2, 2)
    with pytest.raises(ValueError, match="scalar coefficients"):
        ddim_inversion_step(z, z, *map(torch.tensor, pair))


def test_shape_finiteness_and_dtype_errors():
    z = torch.zeros(1, 4, 2, 2)
    for bad in (z[:, :1], torch.full_like(z, float("nan")), z.long()):
        with pytest.raises(ValueError):
            ddim_inversion_step(z, bad, torch.tensor(1.0), torch.tensor(0.5))
    output = ddim_inversion_step(z.half(), z.half(), torch.tensor(1.0), torch.tensor(0.5))
    assert output.dtype == torch.float32
    with pytest.raises(ValueError, match="scalar"):
        ddim_inversion_step(z, z, torch.ones(1), torch.tensor(0.5))


def test_inverter_input_errors_and_failed_model():
    schedule = DDIMSchedule(torch.full((5,), 0.01), (4, 1))
    inverter = DDIMInverter(schedule, lambda z, t, e: z)
    z, e = torch.zeros(1, 4, 2, 2), torch.ones(1, 3, 5)
    for kwargs in ({"guidance_scale": 7.5}, {"source_embeddings": torch.ones(3, 3, 5)},
                   {"source_embeddings": e.long()}, {"latents": torch.full_like(z, float("inf"))}):
        with pytest.raises(ValueError):
            inverter.invert(**({"latents": z, "source_embeddings": e} | kwargs))
    with pytest.raises(ValueError, match="finite"):
        DDIMInverter(schedule, lambda z, t, e: z + float("nan")).invert(z, e)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, z, timestep, encoder_hidden_states):
        self.calls.append((z.dtype, timestep, encoder_hidden_states.dtype, z.device))
        return SimpleNamespace(sample=z * 0 + encoder_hidden_states.mean())


def test_adapter_uses_model_dtype_and_returns_fp32():
    model = TinyUNet().double()
    predictor = UNetNoisePredictor(model)
    result = predictor(torch.ones(1, 4, 2, 2), 3, torch.ones(1, 2, 3))
    assert result.dtype == torch.float32
    assert model.calls[0][:3] == (torch.float64, 3, torch.float64)
    assert not model.training and not model.anchor.requires_grad
    with pytest.raises(ValueError, match="nonnegative"):
        predictor(torch.ones(1, 4, 2, 2), -1, torch.ones(1, 2, 3))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fp16_predictor_fp32_updates_cpu_trajectory():
    model = TinyUNet().cuda().half()
    schedule = DDIMSchedule(torch.full((10,), 0.01), (9, 5, 1))
    result = DDIMInverter(schedule, UNetNoisePredictor(model)).invert(
        torch.ones(1, 4, 2, 2, device="cuda"), torch.ones(1, 2, 3, device="cuda"))
    assert all(z.dtype == torch.float32 and z.device.type == "cpu" for z in result.trajectory.values())
    assert all(call[0] == torch.float16 and call[3].type == "cuda" for call in model.calls)
