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
    plain = pipeline.reconstruct(image, "a colorful gradient", 4, 3, inversion_guidance_scale=3,
                                  inversion_refinement_steps=0)
    assert first.fidelity_check is not None and plain.fidelity_check is None
    # Measure fidelity in latent space against the SAME VAE-encoded image.
    source_latent = first.inversion.initial_latent.to(first.sampling.final_latent.device)
    refined_error = (first.sampling.final_latent - source_latent).square().mean()
    plain_error = (plain.sampling.final_latent - source_latent).square().mean()
    assert refined_error <= plain_error
    if first.fidelity_check.refined_latent_mse > first.fidelity_check.plain_latent_mse:
        assert first.fidelity_check.selected_method == "plain_ddim_fallback"
        assert first.inversion.refinement is None


@pytest.mark.integration
def test_cached_natural_image_refinement_improves_fidelity():
    import torch
    from PIL import Image
    from diffusion_editor.config import load_config
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline

    pipeline = ReconstructionPipeline(SDBackend.load(load_config("configs/default.yaml")))
    with Image.open("assets/examples/astronaut.png") as image:
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        result = pipeline.reconstruct(image, "a portrait of an astronaut wearing a spacesuit", 50)
    check = result.fidelity_check
    assert check.selected_method == "residual_refined_ddim"
    assert check.refined_latent_mse < check.plain_latent_mse / 10
    assert torch.isfinite(result.ddim_reconstruction).all()
