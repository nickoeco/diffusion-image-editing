import pytest
import torch

from test_global_editing import editor  # Reuse the established source/target orchestration fixture.


@pytest.mark.parametrize("preserve", [False, True])
def test_empty_mask_is_vae_or_exact_original(editor, preserve):
    pipeline, _ = editor
    pipeline.vae.spatial_factor = 1
    # A lossy VAE stub makes compositing behavior distinguishable from latent blending.
    pipeline.vae.decode_latents = lambda z: (z / 2 + 0.55).clamp(0, 1)
    image = torch.ones(1, 3, 8, 12) * 0.4
    result = pipeline.edit_local(image, "src", "tgt", torch.zeros(8, 12), 0.5, 1, 3,
                                 preserve_background=preserve, return_trajectory=True)
    expected = image if preserve else result.vae_reconstruction
    torch.testing.assert_close(result.edited_image, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.target_sampling.final_latent, result.inversion.initial_latent, rtol=0, atol=0)


def test_full_mask_matches_global_and_disabled_flags_are_independent(editor):
    pipeline, _ = editor
    pipeline.vae.spatial_factor = 1
    image = torch.ones(1, 3, 8, 12) * 0.5
    global_result = pipeline.edit_global(image, "src", "tgt", 0.5, 1, 3)
    full = pipeline.edit_local(image, "src", "tgt", torch.ones(8, 12), 0.5, 1, 3, feather_radius=2)
    torch.testing.assert_close(full.edited_image, global_result.edited_image, rtol=0, atol=0)
    unmasked = pipeline.edit_local(image, "src", "tgt", torch.zeros(8, 12), 0.5, 1, 3,
                                   latent_blending=False, preserve_background=False)
    torch.testing.assert_close(unmasked.edited_image, global_result.edited_image, rtol=0, atol=0)
    composite_only = pipeline.edit_local(image, "src", "tgt", torch.zeros(8, 12), 0.5, 1, 3,
                                         latent_blending=False, preserve_background=True)
    torch.testing.assert_close(composite_only.edited_image, image, rtol=0, atol=0)


def test_partial_soft_mask_and_zero_strength(editor):
    pipeline, calls = editor
    pipeline.vae.spatial_factor = 1
    image = torch.ones(1, 3, 8, 12) * 0.5
    mask = torch.zeros(8, 12)
    mask[2:6, 4:8] = 0.6
    result = pipeline.edit_local(image, "src", "tgt", mask, 0.5, 1, 3, binary_mask=False)
    assert result.pixel_mask[0, 0, 3, 5] == 0.6
    assert (result.edited_image[:, :, 2:6, 4:8] != image[:, :, 2:6, 4:8]).any()
    background = (result.pixel_mask == 0).expand_as(image)
    torch.testing.assert_close(result.edited_image[background], image[background], rtol=0, atol=0)
    calls.clear()
    zero = pipeline.edit_local(image, "src", "tgt", mask, 0, 1, 3, binary_mask=False)
    assert calls == [] and zero.inversion is None


def test_mask_size_error_precedes_diffusion(editor):
    pipeline, calls = editor
    pipeline.vae.spatial_factor = 1
    with pytest.raises(ValueError, match="shape"):
        pipeline.edit_local(torch.zeros(1, 3, 8, 12), "src", "tgt", torch.ones(7, 12), num_steps=3)
    assert calls == []
