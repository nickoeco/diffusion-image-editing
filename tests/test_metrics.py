import math

import pytest
import torch

from diffusion_editor.evaluation.metrics import reconstruction_metrics


def test_identical_and_known_psnr():
    image = torch.zeros(1, 3, 16, 16)
    identical = reconstruction_metrics(image, image)
    assert math.isinf(identical["psnr_db"]) and identical["ssim"] == 1
    other = reconstruction_metrics(image, torch.ones_like(image) * 0.1)
    assert other["psnr_db"] == pytest.approx(20, abs=1e-5)
    assert 0 < other["ssim"] < 1


def test_metrics_match_skimage_reference():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    reconstructed = (image * 0.9 + 0.02).clamp(0, 1)
    actual = reconstruction_metrics(image, reconstructed)
    a, b = (x[0].permute(1, 2, 0).numpy() for x in (image, reconstructed))
    assert actual["psnr_db"] == pytest.approx(peak_signal_noise_ratio(a, b, data_range=1), abs=1e-6)
    assert actual["ssim"] == pytest.approx(structural_similarity(a, b, channel_axis=-1, data_range=1, win_size=7))


@pytest.mark.parametrize("bad", [torch.zeros(2, 3, 16, 16), torch.zeros(1, 3, 5, 5),
                                    torch.zeros(3, 16, 16), torch.full((1, 3, 16, 16), float("nan"))])
def test_invalid_metric_inputs(bad):
    with pytest.raises(ValueError):
        reconstruction_metrics(torch.zeros(1, 3, 16, 16), bad)
