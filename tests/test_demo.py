"""Offline input/service contracts; HTTP and real weights require explicit flags."""
from dataclasses import replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.config import ProjectConfig
from diffusion_editor.demo.inputs import editor_canvas, normalize_image, resolve_mask
from diffusion_editor.demo.service import DemoRequest, DemoService
from test_reconstruction import FakeVAE, FakeText, FakeTokenizer, FakeUNet, FakeScheduler


@pytest.fixture
def image():
    return Image.fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3))


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion_editor.demo.service.inspect_environment", lambda: {"test": "offline"})
    calls = []
    def load(config):
        calls.append(config)
        return SimpleNamespace(vae=FakeVAE(), tokenizer=FakeTokenizer(), text_encoder=FakeText(),
                               unet=FakeUNet(), scheduler=FakeScheduler(), device="cpu")
    instance = DemoService(ProjectConfig(device="cpu"), tmp_path, backend_loader=load)
    instance.load_calls = calls
    return instance


def test_drawing_uses_alpha_union_and_ignores_composite(image):
    canvas = editor_canvas(image)
    canvas["composite"] = Image.new("RGBA", image.size, "white")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer.putpixel((3, 4), (0, 0, 0, 128))
    canvas["layers"] = [layer, layer.copy()]
    mask = np.asarray(resolve_mask(image, "Draw", None, canvas))
    assert mask[4, 3] == 192
    assert np.count_nonzero(mask) == 1


def test_mask_validation(image):
    with pytest.raises(ValueError, match="No painted"):
        resolve_mask(image, "Draw", None, editor_canvas(image))
    wrong = editor_canvas(Image.new("RGB", image.size))
    with pytest.raises(ValueError, match="different image"):
        resolve_mask(image, "Draw", None, wrong)
    with pytest.raises(ValueError, match="does not match"):
        resolve_mask(image, "Upload", Image.new("L", (2, 2)), None)
    with pytest.raises(ValueError, match="grayscale"):
        resolve_mask(image, "Upload", image, None)
    canvas = editor_canvas(image)
    canvas["layers"] = [Image.new("RGB", image.size)]
    with pytest.raises(ValueError, match="RGBA"):
        resolve_mask(image, "Draw", None, canvas)
    with pytest.raises(ValueError, match="original"):
        normalize_image(None)
    assert normalize_image(image.convert("RGBA")).mode == "RGB"
    assert editor_canvas(None) is None


@pytest.mark.parametrize("field,value", [
    ("mode", "unknown"), ("source_prompt", " "), ("target_prompt", ""),
    ("steps", 1.5), ("steps", 0), ("seed", -1), ("seed", True),
    ("strength", "bad"), ("strength", float("nan")), ("guidance", -1),
    ("feather_radius", "bad"), ("mask_source", "bad"), ("soft_mask", 1),
])
def test_invalid_request(field, value):
    with pytest.raises(ValueError):
        DemoRequest(**{field: value})


def test_lazy_models_validation_before_loading_and_offline_policy(service, image):
    assert not service.load_calls
    with pytest.raises(ValueError, match="Upload a mask"):
        service.run(image, None, None, DemoRequest(steps=3))
    with pytest.raises(ValueError, match="8 and 1024"):
        service.run(Image.new("RGB", (1025, 16)), None, None, DemoRequest(mode="Global edit", steps=3))
    assert not service.load_calls
    with pytest.raises(ValueError, match="local_files_only"):
        DemoService(ProjectConfig(local_files_only=False))


def test_local_run_artifacts_reuse_and_reconstruction(service, image):
    mask = Image.new("L", image.size)
    mask.paste(255, (8, 4, 18, 12))
    progress = []
    request = DemoRequest(steps=3, strength=0.5, show_intermediates=True)
    result = service.run(image, mask, None, request, lambda fraction, text: progress.append(fraction))
    np.testing.assert_array_equal(np.asarray(result.edited)[np.asarray(mask) == 0],
                                  np.asarray(image)[np.asarray(mask) == 0])
    assert result.report["selection"]["reverse_steps"] == 2
    assert result.report["background_metrics"]["final"]["background_mae"] == 0
    assert result.reconstruction is not None and result.control is not None
    assert result.previews and progress[-1] == 1
    with zipfile.ZipFile(result.archive_file) as archive:
        assert {"original.png", "edited.png", "reconstruction.png", "input_mask.png", "report.json", "metrics.csv", "comparison.png"} <= set(archive.namelist())
        report = json.loads(archive.read("report.json"))
        assert report["request"]["seed"] == 42
        assert report["config"]["revision"] == ProjectConfig().revision
        assert report["source_png_sha256"]
        assert report["runtime"]["device"] == "cpu"
    second = service.run(image, None, None, replace(request, mode="Global edit", include_reconstruction=False))
    assert len(service.load_calls) == 1 and not second.report["model_first_load"]
    assert second.reconstruction is None and second.mask is None
    assert Path(second.archive_file).parent != Path(result.archive_file).parent


def test_reconstruction_mode_and_invalid_dimensions(service, image):
    result = service.run(image, None, None, DemoRequest(mode="Reconstruct source", steps=3, refine_reconstruction=False))
    assert result.edited is None and result.control is None and result.reconstruction
    assert result.report["reconstruction_method"] == "plain_ddim"
    assert Path(result.result_file).name == "reconstruction.png"
    with pytest.raises(ValueError, match="multiples of 2"):
        service.run(Image.new("RGB", (17, 24)), None, None, DemoRequest(mode="Global edit", steps=3))


@pytest.mark.demo
def test_http_upload_download_draw_and_errors(service, image, tmp_path):
    pytest.importorskip("gradio")
    from gradio_client import Client, handle_file
    from diffusion_editor.demo.app import create_app
    image_path, mask_path = tmp_path / "input.png", tmp_path / "mask.png"
    image.save(image_path)
    Image.new("L", image.size).save(mask_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(service=service)
    canvas = next(c for c in app.config["components"] if c["type"] == "imageeditor")
    assert canvas["props"]["interactive"] is True and canvas["props"]["visible"] is True
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True,
                   allowed_paths=[str(tmp_path)], quiet=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        args = [handle_file(image_path), handle_file(mask_path), None, "Local edit", "coffee", "pencil",
                3, 0.5, 3, 42, "Upload", False, 0, True, True, 1, "", False, False, False]
        result = client.predict(*args, api_name="/run")
        assert "complete" in result[7] and result[8]["request"]["steps"] == 3
        assert Path(result[9]).is_file() and zipfile.is_zipfile(result[10])
        layer_path = tmp_path / "paint.png"
        layer = Image.new("RGBA", image.size)
        layer.paste((255, 255, 255, 255), (4, 4, 12, 12))
        layer.save(layer_path)
        args[2] = {"background": handle_file(image_path), "layers": [handle_file(layer_path)], "composite": handle_file(image_path)}
        args[10] = "Draw"
        drawn = client.predict(*args, api_name="/run")
        assert drawn[8]["request"]["mask_source"] == "Draw"
        args[2]["layers"] = []
        with pytest.raises(Exception, match="No painted region"):
            client.predict(*args, api_name="/run")
    finally:
        app.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_real_demo_service(tmp_path):
    image = Image.open("assets/examples/coffee.png").convert("RGB").resize((64, 64))
    mask = Image.new("L", image.size)
    mask.paste(255, (16, 16, 48, 48))
    result = DemoService(ProjectConfig(), tmp_path).run(
        image, mask, None, DemoRequest(steps=4, strength=0.5, guidance=3, include_reconstruction=False))
    assert result.report["background_metrics"]["final"]["background_mae"] == 0
    assert result.report["selection"]["reverse_steps"] == 2
    assert zipfile.is_zipfile(result.archive_file)
