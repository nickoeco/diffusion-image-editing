"""Real checkpoint verification; never downloads weights during pytest."""
import pytest


@pytest.mark.integration
def test_real_vae_and_clip_conditioning():
    import torch
    from diffusion_editor.config import load_config
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.models.text_encoder import TextEncoderWrapper
    from diffusion_editor.models.vae import VAEWrapper

    backend = SDBackend.load(load_config("configs/default.yaml"))
    vae = VAEWrapper(backend.vae)
    text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
    image = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    latent = vae.encode_image(image)
    with torch.inference_mode():
        reference = backend.vae.encode((2 * image - 1).to(vae.device)).latent_dist.mode()
    torch.testing.assert_close(latent, reference * backend.vae.config.scaling_factor)
    torch.testing.assert_close(latent, vae.encode_image(image), rtol=0, atol=0)
    reconstructed = vae.decode_latents(latent)
    assert reconstructed.shape == image.shape
    assert torch.isfinite(reconstructed).all()
    pair = text.encode_prompt_pair(["a red apple", "a blue apple"])
    assert pair.shape == (4, backend.tokenizer.model_max_length, backend.text_encoder.config.hidden_size)
    tokens = backend.tokenizer(["", "", "a red apple", "a blue apple"], padding="max_length",
                               max_length=backend.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
    kwargs = ({"attention_mask": tokens.attention_mask.to(backend.device)}
              if getattr(backend.text_encoder.config, "use_attention_mask", False) else {})
    with torch.inference_mode():
        reference = backend.text_encoder(tokens.input_ids.to(backend.device), **kwargs).last_hidden_state
    torch.testing.assert_close(pair, reference, rtol=0, atol=0)
