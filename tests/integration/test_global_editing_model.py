import pytest


@pytest.mark.integration
def test_cached_global_editing_strength_and_prompt_controls():
    import torch
    from diffusion_editor.config import load_config
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
    from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline

    backend = SDBackend.load(load_config("configs/default.yaml"))
    editor = EditingPipeline(backend)
    image = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    source, target = "a red object", "a blue object"
    zero = editor.edit_global(image, source, target, 0, 3, 4)
    assert zero.inversion is None and zero.target_sampling.timesteps == (-1,)
    torch.testing.assert_close(zero.edited_image, zero.vae_reconstruction, rtol=0, atol=0)
    first = editor.edit_global(image, source, target, 0.5, 3, 4, return_trajectory=True)
    repeated = editor.edit_global(image, source, target, 0.5, 3, 4)
    assert first.selection.reverse_steps == 2
    assert len(first.target_sampling.timesteps) == 3
    assert torch.isfinite(first.edited_image).all()
    torch.testing.assert_close(first.edited_image, repeated.edited_image)
    assert not torch.allclose(first.edited_image, first.source_reconstruction)
    full = editor.edit_global(image, source, source, 1, 1, 4)
    reconstruction = ReconstructionPipeline(backend).reconstruct(image, source, 4, 1,
                                                                inversion_refinement_steps=0)
    torch.testing.assert_close(full.edited_image, full.source_reconstruction)
    torch.testing.assert_close(full.edited_image, reconstruction.ddim_reconstruction)
