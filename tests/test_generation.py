"""Text-to-image math, independent demo routing and opt-in HTTP/model checks."""
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

from PIL import Image
import pytest
import torch

from diffusion_editor.config import ProjectConfig
from diffusion_editor.demo.service import DemoService, GenerationRequest
from diffusion_editor.pipelines.generation import TextToImagePipeline
from test_reconstruction import FakeScheduler, FakeText, FakeTokenizer, FakeUNet, FakeVAE


def fake_backend():
    scheduler = FakeScheduler()
    scheduler.init_noise_sigma = 1.0
    return SimpleNamespace(vae=FakeVAE(), tokenizer=FakeTokenizer(), text_encoder=FakeText(),
                           unet=FakeUNet(), scheduler=scheduler, device="cpu")


@pytest.fixture
def generation_service(tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion_editor.demo.service.inspect_environment", lambda: {"test": "synthetic CPU"})
    calls = []
    def load(config):
        calls.append(config)
        return fake_backend()
    service = DemoService(ProjectConfig(device="cpu"), tmp_path / "runs", backend_loader=load)
    service.load_calls = calls
    return service


@pytest.mark.parametrize("guidance", [0, 1, 4])
def test_custom_generation_equation_shapes_and_rng_isolation(guidance):
    backend = fake_backend()
    backend.scheduler.init_noise_sigma = 2.0
    pipeline = TextToImagePipeline(backend)
    before = torch.random.get_rng_state().clone()
    result = pipeline.generate("apple", width=24, height=16, num_steps=3, guidance_scale=guidance, seed=17)
    assert torch.equal(before, torch.random.get_rng_state())
    initial = torch.randn((1, 3, 8, 12), generator=torch.Generator().manual_seed(17)) * 2
    torch.testing.assert_close(result.initial_latent, initial, atol=0, rtol=0)
    alpha = backend.scheduler.alphas_cumprod[6]
    expected = (initial - (1 - alpha).sqrt() * (0.1 * guidance)) / alpha.sqrt()
    torch.testing.assert_close(result.sampling.final_latent, expected, atol=1e-6, rtol=1e-6)
    assert result.image.shape == (1, 3, 16, 24)
    assert result.sampling.timesteps == (6, 2, 0, -1)
    assert result.sampling.trajectory is None and result.image.dtype == torch.float32
    assert torch.isfinite(result.image).all() and 0 <= result.image.min() <= result.image.max() <= 1


@pytest.mark.parametrize("key,value", [("prompt", ""), ("negative_prompt", None), ("width", 65),
    ("height", 2048), ("width", True), ("steps", 0), ("seed", -1), ("guidance", float("nan"))])
def test_generation_request_validation(key, value):
    with pytest.raises(ValueError):
        GenerationRequest(**({"prompt": "apple"} | {key: value}))


def test_generation_service_needs_no_image_inversion_or_vae_encode(generation_service):
    service = generation_service
    service._models()
    service._backend.vae.encode = Mock(side_effect=AssertionError("Generation must not encode an original"))
    service._editor.edit_local = service._editor.edit_global = Mock(side_effect=AssertionError("No editing"))
    service._reconstructor.reconstruct = Mock(side_effect=AssertionError("No source reconstruction"))
    request = GenerationRequest("apple", width=64, height=64, steps=3, seed=123)
    first = service.generate(request)
    second = service.generate(request)
    changed = service.generate(GenerationRequest("apple", width=64, height=64, steps=3, seed=124))
    assert len(service.load_calls) == 1
    assert first.report["initial_noise"]["sha256"] == second.report["initial_noise"]["sha256"]
    assert first.report["output_png_sha256"] == second.report["output_png_sha256"]
    assert first.report["initial_noise"]["sha256"] != changed.report["initial_noise"]["sha256"]
    assert first.report["metrics"] is None and first.image.size == (64, 64)
    assert Path(first.archive_file).parent != Path(second.archive_file).parent
    with zipfile.ZipFile(first.archive_file) as archive:
        assert set(archive.namelist()) == {"generated.png", "report.json"}
        report = json.loads(archive.read("report.json"))
        assert report["mode"] == "txt2img" and report["request"]["seed"] == 123
        assert report["sampling_timesteps"] == [6, 2, 0, -1]


def test_invalid_pipeline_grid_and_prior():
    backend = fake_backend()
    pipeline = TextToImagePipeline(backend)
    with pytest.raises(ValueError, match="multiples"):
        pipeline.generate("apple", width=17, height=16, num_steps=3)
    backend.scheduler.init_noise_sigma = float("nan")
    with pytest.raises(ValueError, match="init_noise_sigma"):
        pipeline.generate("apple", width=16, height=16, num_steps=3)


@pytest.mark.demo
def test_txt2img_http_api_and_downloads(generation_service):
    pytest.importorskip("gradio")
    from gradio_client import Client
    from diffusion_editor.demo.app import create_app
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(service=generation_service)
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True,
                   allowed_paths=[str(generation_service.output_dir)], quiet=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        result = client.predict("apple", "", 64, 64, 3, 4, 42, api_name="/txt2img")
        assert Image.open(result[0]).size == (64, 64)
        assert "Text to image complete" in result[1]
        assert Path(result[2]).is_file() and zipfile.is_zipfile(result[3])
        assert result[4]["request"]["prompt"] == "apple"
        assert result[4]["method"] == "custom_ddim_eta0_cfg"
        with pytest.raises(Exception, match="generation prompt"):
            client.predict("", "", 64, 64, 3, 4, 42, api_name="/txt2img")
    finally:
        app.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_cached_model_text_to_image(tmp_path):
    service = DemoService(ProjectConfig(), tmp_path)
    result = service.generate(GenerationRequest("a red apple on a table", width=64, height=64, steps=4))
    assert result.image.size == (64, 64)
    assert len(result.report["sampling_timesteps"]) == 5
    assert result.report["runtime"]["vae_dtype"] == "torch.float32"
    assert zipfile.is_zipfile(result.archive_file)
