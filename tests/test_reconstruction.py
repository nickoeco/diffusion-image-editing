from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
from diffusion_editor.utils.visualization import save_image_grid


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=7, prediction_type="epsilon", clip_sample=False)
        self.betas = torch.full((7,), 0.02)
        self.alphas_cumprod = (1 - self.betas).cumprod(0)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls()

    def set_timesteps(self, steps, device):
        assert steps == 3
        self.timesteps = torch.tensor([6, 2, 0])


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=0.3, block_out_channels=(8, 16), latent_channels=3)

    def encode(self, image):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: F.avg_pool2d(image, 2)))

    def decode(self, z):
        return SimpleNamespace(sample=F.interpolate(z, scale_factor=2, mode="nearest"))


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **kwargs):
        ids = [[int(bool(p))] * 4 for p in prompts]
        return {"input_ids": torch.tensor(ids) if kwargs.get("return_tensors") else ids}


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=2, use_attention_mask=False)

    def forward(self, ids):
        return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1).expand(-1, -1, 2))


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, z, t, encoder_hidden_states):
        eps = encoder_hidden_states[:, 0, 0].reshape(-1, 1, 1, 1) * 0.1
        return SimpleNamespace(sample=torch.zeros_like(z) + eps)


def make_pipeline():
    return ReconstructionPipeline(SimpleNamespace(vae=FakeVAE(), tokenizer=FakeTokenizer(),
                                  text_encoder=FakeText(), unet=FakeUNet(), scheduler=FakeScheduler(), device="cpu"))


@pytest.mark.parametrize("scale", [0, 1, 5])
def test_end_to_end_same_prompt_and_shared_schedule(scale):
    pipeline = make_pipeline()
    image = torch.linspace(0, 1, 2 * 3 * 16 * 24).reshape(2, 3, 16, 24)
    result = pipeline.reconstruct(image, ["red", "blue"], 3, scale, inversion_guidance_scale=scale,
                                  return_trajectory=True)
    assert result.ddim_reconstruction.shape == image.shape
    assert result.inversion.timesteps == tuple(reversed(result.sampling.timesteps))
    torch.testing.assert_close(result.vae_reconstruction, result.ddim_reconstruction, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(result.ddim_reconstruction).all()
    assert result.sampling.trajectory[-1].shape == (2, 3, 8, 12)


def test_prompt_batch_mismatch():
    with pytest.raises(ValueError, match="batch sizes"):
        make_pipeline().reconstruct(torch.zeros(2, 3, 16, 16), "single", 3)


def test_labelled_grid(tmp_path):
    from PIL import Image
    path = tmp_path / "grid.png"
    save_image_grid([torch.zeros(1, 3, 16, 24)] * 3, ["Original", "VAE", "DDIM"], path)
    with Image.open(path) as image:
        assert image.size == (72, 40)
    with pytest.raises(ValueError):
        save_image_grid([], [], path)
