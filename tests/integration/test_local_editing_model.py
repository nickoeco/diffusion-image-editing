import pytest


@pytest.mark.integration
def test_real_local_masks_and_background_preservation():
    import torch
    from diffusion_editor.config import load_config
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.editing_pipeline import EditingPipeline

    editor = EditingPipeline(SDBackend.load(load_config("configs/default.yaml")))
    image = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    options = dict(strength=0.5, guidance_scale=3, num_steps=4, return_trajectory=True)
    empty = editor.edit_local(image, "a red object", "a blue object", torch.zeros(64, 64), **options)
    torch.testing.assert_close(empty.edited_image.cpu(), image, rtol=0, atol=0)
    torch.testing.assert_close(empty.target_sampling.final_latent.cpu(), empty.inversion.initial_latent, rtol=0, atol=0)
    global_result = editor.edit_global(image, "a red object", "a blue object", **options)
    full = editor.edit_local(image, "a red object", "a blue object", torch.ones(64, 64), **options)
    torch.testing.assert_close(full.edited_image, global_result.edited_image, rtol=0, atol=0)
    mask = torch.zeros(64, 64)
    mask[16:48, 16:48] = 0.7
    partial = editor.edit_local(image, "a red object", "a blue object", mask, binary_mask=False,
                                feather_radius=1, **options)
    background = (partial.pixel_mask == 0).expand_as(image)
    torch.testing.assert_close(partial.edited_image.cpu()[background], image[background], rtol=0, atol=0)
    assert torch.isfinite(partial.edited_image).all()
    repeated = editor.edit_local(image, "a red object", "a blue object", mask, binary_mask=False,
                                 feather_radius=1, **options)
    torch.testing.assert_close(partial.edited_image, repeated.edited_image)
