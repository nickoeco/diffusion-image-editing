import pytest
import torch

from diffusion_editor.diffusion.inversion import DDIMInverter, ddim_inversion_step
from diffusion_editor.diffusion.refinement import refine_inversion_step
from diffusion_editor.diffusion.sampler import DDIMSampler, ddim_sampling_step
from diffusion_editor.diffusion.schedule import DDIMSchedule


def test_affine_predictor_matches_analytic_inverse_and_reduces_residual():
    source = torch.linspace(-0.5, 0.5, 32).reshape(1, 4, 2, 4)
    embeddings = torch.ones(1, 2, 3)
    t, s = torch.tensor(0.9), torch.tensor(0.6)
    calls = []
    def predictor(z, timestep, e):
        calls.append(timestep)
        return 0.2 * z + 0.1
    initial = ddim_inversion_step(source, predictor(source, 1, embeddings), t, s)
    result, info = refine_inversion_step(source, initial, t, s, 8, predictor, embeddings, 1,
                                         max_iterations=30, tolerance=1e-7)
    a = (t / s).sqrt()
    b = (1 - t).sqrt() - a * (1 - s).sqrt()
    exact = (source - b * 0.1) / (a + b * 0.2)
    torch.testing.assert_close(result, exact, atol=1e-6, rtol=1e-6)
    assert info.final_rms < info.initial_rms and info.converged
    assert calls[0] == 1 and all(timestep == 8 for timestep in calls[1:])
    recovered = ddim_sampling_step(result, predictor(result, 8, embeddings), s, t)
    torch.testing.assert_close(recovered, source, atol=1e-6, rtol=1e-6)


def test_keeps_best_evaluated_iterate_when_correction_is_unstable():
    # Here the local fixed-point iteration diverges; do not return the final proposal.
    source = torch.ones(1, 4, 2, 2)
    initial = torch.zeros_like(source)
    e = torch.ones(1, 2, 3)
    predict = lambda z, t, e: z * 10
    result, info = refine_inversion_step(source, initial, torch.tensor(1.), torch.tensor(0.5),
                                         5, predict, e, 1, max_iterations=8, tolerance=1e-8)
    assert info.evaluations == 8 and not info.converged
    assert info.final_rms <= info.initial_rms
    torch.testing.assert_close(result, initial, rtol=0, atol=0)


def test_early_stop_and_one_evaluation_return_initial():
    z = torch.zeros(1, 4, 2, 2)
    predictor = lambda z, t, e: z * 0
    result, info = refine_inversion_step(z, z, torch.tensor(1.), torch.tensor(0.8), 3,
                                         predictor, torch.ones(1, 2, 3), 1, max_iterations=12)
    assert info.evaluations == 1 and info.converged
    torch.testing.assert_close(result, z)
    initial = torch.ones_like(z)
    result, info = refine_inversion_step(z, initial, torch.tensor(1.), torch.tensor(0.8), 3,
                                         predictor, torch.ones(1, 2, 3), 1, max_iterations=1)
    torch.testing.assert_close(result, initial)
    assert not info.converged


@pytest.mark.parametrize("kwargs", [{"refinement_steps": -1}, {"refinement_steps": True},
                                     {"refinement_steps": 1.5}, {"refinement_tolerance": 0},
                                     {"refinement_tolerance": float("nan")}])
def test_invalid_refinement_configuration(kwargs):
    schedule = DDIMSchedule(torch.full((5,), 0.01), (4, 1))
    with pytest.raises(ValueError):
        DDIMInverter(schedule, lambda z, t, e: z).invert(torch.zeros(1, 4, 2, 2), torch.ones(1, 2, 3), **kwargs)


def test_source_dependent_noise_roundtrip_improves_without_changing_grid():
    schedule = DDIMSchedule(torch.linspace(0.01, 0.05, 20), (19, 12, 3, 0))
    source = torch.linspace(-0.5, 0.5, 32).reshape(1, 4, 2, 4)
    e = torch.ones(1, 2, 3)
    predictor = lambda z, t, e: 0.3 * z + 0.01 * t
    inverter = DDIMInverter(schedule, predictor)
    plain = inverter.invert(source, e)
    refined = inverter.invert(source, e, refinement_steps=30, refinement_tolerance=1e-7)
    sampler = DDIMSampler(schedule, predictor)
    plain_clean = sampler.sample(plain.final_latent, e).final_latent
    refined_clean = sampler.sample(refined.final_latent, e).final_latent
    assert refined.timesteps == plain.timesteps
    assert plain.refinement is None and len(refined.refinement) == 4
    assert (refined_clean - source).square().mean() < (plain_clean - source).square().mean() / 100
    torch.testing.assert_close(refined_clean, source, atol=1e-6, rtol=1e-6)
