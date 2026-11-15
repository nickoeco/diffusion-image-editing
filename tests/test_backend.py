from dataclasses import replace
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from diffusion_editor.config import ProjectConfig
from diffusion_editor.models.sd_backend import SDBackend, resolve_runtime


def fake_torch(cuda):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           float16="fp16", float32="fp32")


@pytest.mark.parametrize("cuda,config,expected", [
    (False, ProjectConfig(), ("cpu", "fp32")),
    (True, ProjectConfig(), ("cuda", "fp16")),
    (True, ProjectConfig(device="cpu"), ("cpu", "fp32")),
    (True, ProjectConfig(precision="fp32"), ("cuda", "fp32")),
])
def test_runtime(cuda, config, expected):
    assert resolve_runtime(config, fake_torch(cuda)) == expected


def test_unavailable_cuda():
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_runtime(ProjectConfig(device="cuda"), fake_torch(False))
    with pytest.raises(ValueError, match="CPU"):
        resolve_runtime(ProjectConfig(precision="fp16"), fake_torch(False))


@pytest.fixture
def loaders(monkeypatch):
    factories = {name: MagicMock() for name in
                 ("AutoencoderKL", "DDIMScheduler", "UNet2DConditionModel", "CLIPTextModel", "CLIPTokenizer")}
    factories["DDIMScheduler"].from_pretrained.return_value.config.prediction_type = "epsilon"
    factories["AutoencoderKL"].from_pretrained.return_value.config.latent_channels = 4
    factories["UNet2DConditionModel"].from_pretrained.return_value.config.in_channels = 4
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(**{k: factories[k] for k in
                        ("AutoencoderKL", "DDIMScheduler", "UNet2DConditionModel")}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(**{k: factories[k] for k in
                        ("CLIPTextModel", "CLIPTokenizer")}))
    return factories


@pytest.mark.parametrize("offline", [True, False])
def test_loading_contract(loaders, offline):
    config = replace(ProjectConfig(), local_files_only=offline)
    backend = SDBackend.load(config)
    assert backend.device == "cuda"
    for name, factory in loaders.items():
        kwargs = factory.from_pretrained.call_args.kwargs
        assert kwargs["revision"] == config.revision
        assert kwargs["local_files_only"] is offline
        assert kwargs["cache_dir"] == config.cache_dir
        assert "torch_dtype" not in kwargs if name in ("CLIPTokenizer", "DDIMScheduler") else True
    for model in (backend.vae, backend.text_encoder, backend.unet):
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with("cuda")
    assert loaders["AutoencoderKL"].from_pretrained.call_args.kwargs["torch_dtype"] == "fp32"
    assert "variant" not in loaders["AutoencoderKL"].from_pretrained.call_args.kwargs
    assert loaders["UNet2DConditionModel"].from_pretrained.call_args.kwargs["variant"] == "fp16"


def test_reject_other_prediction_type(loaders):
    loaders["DDIMScheduler"].from_pretrained.return_value.config.prediction_type = "v_prediction"
    with pytest.raises(ValueError, match="epsilon"):
        SDBackend.load(ProjectConfig())
    loaders["AutoencoderKL"].from_pretrained.assert_not_called()


def test_reject_inpainting_channels(loaders):
    loaders["UNet2DConditionModel"].from_pretrained.return_value.config.in_channels = 9
    with pytest.raises(ValueError, match="channels"):
        SDBackend.load(ProjectConfig())


def test_cache_miss_is_not_hidden(loaders):
    loaders["DDIMScheduler"].from_pretrained.side_effect = OSError("cache miss")
    with pytest.raises(OSError, match="cache miss"):
        SDBackend.load(ProjectConfig())


def test_cpu_loader_uses_fp32(loaders, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False))
    backend = SDBackend.load(ProjectConfig())
    assert (backend.device, backend.dtype) == ("cpu", "fp32")
    for name in ("AutoencoderKL", "CLIPTextModel", "UNet2DConditionModel"):
        kwargs = loaders[name].from_pretrained.call_args.kwargs
        assert kwargs["torch_dtype"] == "fp32"
        assert "variant" not in kwargs


def test_missing_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="Model dependencies missing"):
        SDBackend.load(ProjectConfig())
