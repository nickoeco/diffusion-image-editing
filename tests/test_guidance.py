import pytest
import torch

from diffusion_editor.diffusion.guidance import classifier_free_guidance, predict_guided_noise


@pytest.mark.parametrize("scale", [0, 1, 7.5])
def test_cfg_equation_and_endpoints(scale):
    u = torch.full((2, 4, 2, 3), 0.2, dtype=torch.float16)
    c = torch.full_like(u, 0.7)
    result = classifier_free_guidance(u, c, scale)
    expected = u.float() + scale * (c.float() - u.float())
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.float32
    if scale in (0, 1):
        torch.testing.assert_close(result, u.float() if scale == 0 else c.float(), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [-1, float("inf"), float("nan"), True, "7.5"])
def test_invalid_scales(scale):
    z = torch.zeros(1, 4, 2, 2)
    with pytest.raises(ValueError, match="scale"):
        classifier_free_guidance(z, z, scale)


def test_batched_unet_order_and_single_call():
    z = torch.stack((torch.zeros(4, 2, 2), torch.ones(4, 2, 2)))
    e = torch.tensor([1., 2., 4., 7.]).reshape(4, 1, 1)
    calls = []
    def predictor(latent, timestep, embeddings):
        calls.append((latent.clone(), embeddings.clone(), timestep))
        return latent * 0 + embeddings[:, 0, 0].reshape(-1, 1, 1, 1)
    result = predict_guided_noise(predictor, z, 8, e, 3)
    torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([10., 17.]))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.cat((z, z)))
    torch.testing.assert_close(calls[0][1], e)
    assert calls[0][2] == 8


def test_invalid_batches_and_predictions():
    z = torch.zeros(1, 4, 2, 2)
    predictor = lambda z, t, e: z
    for e, scale in ((torch.ones(3, 2, 2), 1), (torch.ones(1, 2, 2), 5)):
        with pytest.raises(ValueError):
            predict_guided_noise(predictor, z, 1, e, scale)
    with pytest.raises(ValueError, match="invalid shape"):
        predict_guided_noise(lambda z, t, e: z[:1], z, 1, torch.ones(2, 2, 2), 5)
    with pytest.raises(ValueError, match="finite"):
        classifier_free_guidance(z, z + float("nan"), 1)
