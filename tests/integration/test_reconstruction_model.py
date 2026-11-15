import pytest


@pytest.mark.integration
def test_cached_pretrained_reconstruction_with_cfg_and_repeatability():
    import torch
    from diffusion_editor.config import load_config
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline

    pipeline = ReconstructionPipeline(SDBackend.load(load_config("configs/default.yaml")))
    image = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    first = pipeline.reconstruct(image, "a colorful gradient", 4, 3,
                                 inversion_guidance_scale=3, return_trajectory=True)
    repeated = pipeline.reconstruct(image, "a colorful gradient", 4, 3, inversion_guidance_scale=3)
    assert first.ddim_reconstruction.shape == image.shape
    assert first.inversion.timesteps == tuple(reversed(first.sampling.timesteps))
    assert len(first.sampling.trajectory) == 5
    assert torch.isfinite(first.ddim_reconstruction).all()
    torch.testing.assert_close(first.ddim_reconstruction, repeated.ddim_reconstruction)
