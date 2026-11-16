import numpy as np
from PIL import Image
import pytest
import torch

from diffusion_editor.editing.mask import prepare_mask


def test_binary_alignment_threshold_and_batch_broadcast():
    mask = torch.zeros(8, 12)
    mask[4:, 4:8] = 0.5
    mask[:4, :4] = 0.49
    prepared = prepare_mask(mask, (8, 12), (2, 3), batch_size=2)
    assert prepared.pixel.shape == (2, 1, 8, 12)
    expected = torch.tensor([[[[0., 0., 0.], [0., 1., 0.]]]]).expand(2, -1, -1, -1)
    torch.testing.assert_close(prepared.latent, expected)
    assert mask[0, 0] == 0.49  # Input was not thresholded in place.


def test_soft_mask_resize_and_pil_conversion():
    soft = torch.arange(12).float().reshape(1, 12).expand(8, -1) / 11
    prepared = prepare_mask(soft, (8, 12), (2, 3), binary=False)
    torch.testing.assert_close(prepared.latent[0, 0, 0], torch.tensor([1.5, 5.5, 9.5]) / 11)
    pil = Image.fromarray(np.full((8, 12), 255, dtype=np.uint8))
    assert torch.all(prepare_mask(pil, (8, 12), (2, 3)).latent == 1)


def test_feathering_expands_support_and_preserves_batch_constants():
    mask = torch.zeros(1, 1, 24, 32)
    mask[:, :, 8:16, 12:20] = 1
    prepared = prepare_mask(mask, (24, 32), (3, 4), feather_radius=2)
    assert 0 < prepared.pixel[0, 0, 7, 15] < 1
    assert ((prepared.pixel > 0) & (prepared.pixel < 1)).any()
    assert prepared.pixel[0, 0, 0, 0] == 0
    constants = torch.cat((torch.zeros_like(mask), torch.ones_like(mask)))
    result = prepare_mask(constants, (24, 32), (3, 4), batch_size=2, feather_radius=2)
    torch.testing.assert_close(result.pixel, constants, rtol=0, atol=0)
    assert torch.all(result.latent[0] == 0) and torch.all(result.latent[1] == 1)


@pytest.mark.parametrize("bad", [torch.zeros(3, 8, 12), torch.zeros(8, 11), torch.zeros(2, 2, 8, 12),
                                    torch.full((8, 12), 1.1), torch.full((8, 12), float("nan"))])
def test_bad_masks(bad):
    with pytest.raises(ValueError):
        prepare_mask(bad, (8, 12), (2, 3))


def test_ambiguous_integer_or_color_masks_rejected():
    with pytest.raises(TypeError):
        prepare_mask(torch.ones(8, 12, dtype=torch.int64), (8, 12), (2, 3))
    with pytest.raises(ValueError, match="single-channel"):
        prepare_mask(Image.new("RGB", (12, 8)), (8, 12), (2, 3))
    assert torch.all(prepare_mask(torch.ones(8, 12, dtype=torch.bool), (8, 12), (2, 3)).latent == 1)


@pytest.mark.parametrize("radius", [-1, float("nan"), float("inf"), 100, True])
def test_invalid_feather_radius(radius):
    with pytest.raises(ValueError, match="feather"):
        prepare_mask(torch.ones(8, 12), (8, 12), (2, 3), feather_radius=radius)


def test_tiny_mask_disappearance_is_explicit():
    mask = torch.zeros(8, 8)
    mask[7, 7] = 1
    with pytest.warns(UserWarning, match="disappears"):
        result = prepare_mask(mask, (8, 8), (1, 1))
    assert not torch.any(result.latent)
