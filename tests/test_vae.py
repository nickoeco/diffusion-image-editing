from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.utils.image_utils import image_to_tensor, save_comparison, tensor_to_images


class TinyVAE(nn.Module):
    """A deterministic shape/scaling oracle, not a real reconstruction model."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=0.37, block_out_channels=(8, 16), latent_channels=4)
        self.posterior = None

    def encode(self, image):
        self.encoded_input = image.clone()
        pooled = F.avg_pool2d(image, 2)
        latent = torch.cat((pooled, pooled[:, :1]), dim=1)
        self.posterior = SimpleNamespace(mode=Mock(return_value=latent), sample=Mock())
        return SimpleNamespace(latent_dist=self.posterior)

    def decode(self, latent):
        self.decoded_input = latent.clone()
        return SimpleNamespace(sample=F.interpolate(latent[:, :3], scale_factor=2, mode="nearest"))


def test_normalization_scaling_and_inverse():
    model = TinyVAE()
    wrapper = VAEWrapper(model)
    image = torch.ones(2, 3, 12, 16, requires_grad=True) * 0.75
    latent = wrapper.encode_image(image)
    torch.testing.assert_close(model.encoded_input, torch.full_like(image, 0.5))
    torch.testing.assert_close(latent, torch.full((2, 4, 6, 8), 0.5 * 0.37))
    assert not latent.requires_grad
    model.posterior.mode.assert_called_once()
    model.posterior.sample.assert_not_called()
    output = wrapper.decode_latents(latent)
    torch.testing.assert_close(model.decoded_input, torch.full_like(latent, 0.5))
    torch.testing.assert_close(output, image)
    assert output.dtype == torch.float32
    assert not model.training and not model.anchor.requires_grad
    torch.testing.assert_close(wrapper.encode_image(image), latent, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [torch.zeros(3, 8, 8), torch.zeros(1, 1, 8, 8),
                                    torch.zeros(1, 3, 7, 8), torch.full((1, 3, 8, 8), float("nan")),
                                    torch.ones(1, 3, 8, 8) * 1.1, torch.zeros(0, 3, 8, 8)])
def test_invalid_image(bad):
    with pytest.raises(ValueError):
        VAEWrapper(TinyVAE()).encode_image(bad)


def test_integer_and_nonimage_rejected():
    with pytest.raises(TypeError):
        image_to_tensor(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))
    with pytest.raises(TypeError):
        image_to_tensor("image.png")


@pytest.mark.parametrize("bad", [torch.zeros(4, 4, 4), torch.zeros(1, 3, 4, 4),
                                    torch.full((1, 4, 4, 4), float("inf")),
                                    torch.zeros(1, 4, 4, 4, dtype=torch.int64)])
def test_invalid_latents(bad):
    with pytest.raises(ValueError):
        VAEWrapper(TinyVAE()).decode_latents(bad)


def test_pil_rgb_conversion_and_pixel_roundtrip(tmp_path):
    pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    image = Image.fromarray(pixels)
    tensor = image_to_tensor(image)
    assert tensor.shape == (1, 3, 8, 12)
    np.testing.assert_array_equal(np.asarray(tensor_to_images(tensor)[0]), pixels)
    assert image_to_tensor(image.convert("L")).shape == tensor.shape
    wrapper = VAEWrapper(TinyVAE())
    assert wrapper.encode_image(image).shape == (1, 4, 4, 6)
    target = tmp_path / "comparison.png"
    save_comparison(tensor, tensor, target)
    with Image.open(target) as saved:
        assert saved.size == (24, 32)


def test_config_and_output_validation():
    model = TinyVAE()
    model.config.scaling_factor = 0
    with pytest.raises(ValueError, match="scaling_factor"):
        VAEWrapper(model)
    with pytest.raises(ValueError, match="FP32"):
        VAEWrapper(TinyVAE().half())
    model = TinyVAE()
    wrapper = VAEWrapper(model)
    model.decode = lambda z: SimpleNamespace(sample=torch.full((1, 3, 8, 8), float("nan")))
    with pytest.raises(RuntimeError, match="decoder"):
        wrapper.decode_latents(torch.zeros(1, 4, 4, 4))


def test_endpoints_dtype_and_decode_clamping():
    model = TinyVAE()
    wrapper = VAEWrapper(model)
    pixels = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8))).double()
    latent = wrapper.encode_image(pixels)
    assert latent.dtype == torch.float32
    torch.testing.assert_close(model.encoded_input[0], torch.full((3, 8, 8), -1.0))
    torch.testing.assert_close(model.encoded_input[1], torch.ones(3, 8, 8))
    torch.testing.assert_close(wrapper.decode_latents(latent * 10), pixels.float())


def test_exif_orientation_preserved():
    image = Image.new("RGB", (12, 8), "red")
    image.getexif()[274] = 6
    assert image_to_tensor(image).shape == (1, 3, 12, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_vae_cuda_cpu_input_and_fp16_latent():
    wrapper = VAEWrapper(TinyVAE().cuda())
    latent = wrapper.encode_image(torch.zeros(1, 3, 8, 8))
    assert latent.device.type == "cuda" and latent.dtype == torch.float32
    image = wrapper.decode_latents(latent.half().cpu())
    assert image.device.type == "cuda" and image.dtype == torch.float32
