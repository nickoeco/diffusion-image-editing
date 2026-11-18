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
from diffusion_editor.demo.image_fit import prepare_upload
from diffusion_editor.demo.service import DemoRequest, DemoService
from diffusion_editor.demo.ui_state import CONTROL, DDIM, EDIT, VAE, default_outputs, layout_css, output_choices, validate_outputs
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
        service.run(Image.new("RGB", (1025, 16)), None, None, DemoRequest(mode="Global edit", steps=3, auto_fit=False))
    assert not service.load_calls
    with pytest.raises(ValueError, match="local_files_only"):
        DemoService(ProjectConfig(local_files_only=False))


def test_local_run_artifacts_reuse_and_reconstruction(service, image):
    mask = Image.new("L", image.size)
    mask.paste(255, (8, 4, 18, 12))
    progress = []
    request = DemoRequest(steps=3, strength=0.5, show_intermediates=True, include_reconstruction=True,
                          include_vae_reconstruction=True, include_source_control=True)
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
        service.run(Image.new("RGB", (17, 24)), None, None, DemoRequest(mode="Global edit", steps=3, auto_fit=False))


def test_auto_fit_vae_only_records_original_and_canvas(service):
    image = Image.new("RGB", (1031, 31), "red")
    result = service.run(image, None, None, DemoRequest(include_edit=False, include_vae_reconstruction=True))
    assert result.original.size == result.vae.size == (1024, 32)
    info = result.report["preprocessing"]
    assert info["source_size"] == [1031, 31] and info["content_size"] == [1024, 31]
    assert info["padding_bottom"] == 1 and info["padding_right"] == 0
    assert result.report["metric_domain"] == "fitted_model_canvas_including_padding"
    with zipfile.ZipFile(result.archive_file) as archive:
        assert "uploaded_original.png" in archive.namelist()
        with archive.open("uploaded_original.png") as stream:
            assert Image.open(stream).size == image.size


def test_auto_fit_local_original_mask_and_preview_provenance(service):
    image = Image.new("RGB", (17, 13), "red")
    prepared = prepare_upload(image)
    mask = Image.new("L", image.size)
    mask.paste(255, (4, 4, 12, 10))
    request = DemoRequest(steps=3, strength=0.5)
    direct = service.run(image, mask, None, request)
    preview = service.run(prepared.image, mask, None, request, input_state=prepared)
    np.testing.assert_array_equal(np.asarray(direct.edited), np.asarray(preview.edited))
    fitted_mask = prepared.fit.mask(mask, binary=True)[0]
    preserved = np.asarray(fitted_mask) == 0
    np.testing.assert_array_equal(np.asarray(direct.edited)[preserved], np.asarray(prepared.image)[preserved])
    assert preview.report["preprocessing"]["source_size"] == [17, 13]
    with zipfile.ZipFile(preview.archive_file) as archive:
        assert {"mask_before_fit.png", "input_mask.png", "uploaded_original.png"} <= set(archive.namelist())


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
                3, 0.5, 3, 42, "Upload", False, 0, True, True, 1, "", [EDIT], False, False, True]
        result = client.predict(*args, api_name="/run")
        assert "complete" in result[7] and result[8]["request"]["steps"] == 3
        assert Path(result[9]).is_file() and zipfile.is_zipfile(result[10])
        assert result[8]["computed_outputs"] == [EDIT]
        assert result[8]["source_timesteps"] is None and result[8]["reconstruction_timesteps"] is None
        operation = client.predict("Global edit", api_name="/change_operation")
        assert "#local-mask-panel" in operation[1] and "display:none" in operation[1]
        uploaded = client.predict(handle_file(image_path), "Global edit", [EDIT], True, api_name="/upload_image")
        assert uploaded[0] and uploaded[2] is None
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
        args[17] = [VAE]
        args[4] = args[5] = ""
        args[1] = args[2] = None
        vae_only = client.predict(*args, api_name="/run")
        assert vae_only[8]["computed_outputs"] == [VAE]
        assert vae_only[8]["selection"] is None
        odd_path = tmp_path / "odd.png"
        Image.new("RGB", (17, 13), "red").save(odd_path)
        fitted = client.predict(handle_file(odd_path), "Local edit", [VAE], True, api_name="/upload_image")
        assert Image.open(fitted[0]).size == (24, 16)
        assert "upload 17 x 13" in fitted[3] and "canvas/output 24 x 16" in fitted[3]
        args[0] = handle_file(fitted[0])
        odd_result = client.predict(*args, api_name="/run")
        assert odd_result[8]["preprocessing"]["source_size"] == [17, 13]
        assert Image.open(odd_result[9]).size == (24, 16)
        strict = client.predict(handle_file(fitted[0]), "Local edit", [VAE], False, api_name="/fit_image")
        assert Image.open(strict[0]).size == (17, 13) and strict[1]["layers"] == []
        restored = client.predict(handle_file(strict[0]), "Local edit", [VAE], True, api_name="/fit_image")
        assert "upload 17 x 13" in restored[3] and Image.open(restored[0]).size == (24, 16)
    finally:
        app.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_real_demo_service(tmp_path):
    image = Image.open("assets/examples/coffee.png").convert("RGB").resize((64, 64))
    mask = Image.new("L", image.size)
    mask.paste(255, (16, 16, 48, 48))
    from diffusion_editor.evaluation.artifacts import count_unet
    service = DemoService(ProjectConfig(), tmp_path)
    service._models()
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, mask, None, DemoRequest(steps=4, strength=0.5, guidance=3))
    assert budget["unet_calls"] == 6
    assert result.vae is None and result.control is None and result.reconstruction is None
    assert result.report["background_metrics"]["final"]["background_mae"] == 0
    assert result.report["selection"]["reverse_steps"] == 2
    assert zipfile.is_zipfile(result.archive_file)
    odd = image.crop((0, 0, 57, 49))
    with count_unet(service._backend.unet) as budget:
        vae = service.run(odd, None, None, DemoRequest(include_edit=False, include_vae_reconstruction=True))
    assert budget["unet_calls"] == 0 and vae.vae is not None
    assert vae.vae.size == (64, 56)
    assert vae.report["preprocessing"]["source_size"] == [57, 49]


def test_output_rules_and_mode_specific_controls():
    assert default_outputs("Local edit") == [EDIT]
    assert default_outputs("Reconstruct source") == [DDIM]
    assert output_choices("Reconstruct source") == [VAE, DDIM]
    assert "#local-mask-panel" not in layout_css("Local edit", [EDIT])
    assert "#local-mask-panel" in layout_css("Global edit", [EDIT])
    assert "#local-mask-panel" in layout_css("Reconstruct source", [DDIM])
    assert "#local-mask-panel" in layout_css("Local edit", [VAE])
    with pytest.raises(ValueError, match="at least one"):
        validate_outputs("Global edit", [])
    with pytest.raises(ValueError, match="valid"):
        validate_outputs("Reconstruct source", [EDIT])
    with pytest.raises(ValueError, match="at least one"):
        DemoRequest(include_edit=False)


def test_edit_only_skips_control_ddim_and_vae_decode(service, image):
    from unittest.mock import Mock
    from diffusion_editor.evaluation.artifacts import count_unet
    service._models()
    vae = service._backend.vae
    vae.decode = Mock(wraps=vae.decode)
    service._reconstructor.reconstruct = Mock(side_effect=AssertionError("Unrequested DDIM work"))
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, None, None, DemoRequest(mode="Global edit", steps=3, strength=0.5))
    assert budget["unet_calls"] == 5  # Three inversion calls + two target calls.
    assert vae.decode.call_count == 1
    assert result.vae is None and result.control is None and result.reconstruction is None
    assert result.report["source_timesteps"] is None
    with zipfile.ZipFile(result.archive_file) as archive:
        assert not {"vae.png", "control.png", "reconstruction.png"} & set(archive.namelist())
    reference = service._editor.edit_global(image, "coffee", "pencil", 0.5, 7.5, 3)
    from diffusion_editor.utils.image_utils import tensor_to_images
    np.testing.assert_array_equal(np.asarray(result.edited), np.asarray(tensor_to_images(reference.edited_image)[0]))


def test_vae_only_skips_prompts_mask_and_all_diffusion(service, image):
    from unittest.mock import Mock
    from diffusion_editor.evaluation.artifacts import count_unet
    service._models()
    vae = service._backend.vae
    vae.decode = Mock(wraps=vae.decode)
    service._editor.edit_local = service._editor.edit_global = Mock(side_effect=AssertionError("Unrequested editing"))
    service._reconstructor.reconstruct = Mock(side_effect=AssertionError("Unrequested DDIM work"))
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, None, None, DemoRequest(source_prompt="", target_prompt="", steps=3,
                            include_edit=False, include_vae_reconstruction=True))
    assert budget["unet_calls"] == 0
    assert vae.decode.call_count == 1
    assert result.vae and result.edited is None and result.reconstruction is None and result.mask is None
    assert result.report["computed_outputs"] == [VAE]
    assert Path(result.result_file).name == "vae.png"


def test_ddim_only_skips_editing_and_vae_baseline(service, image):
    from unittest.mock import Mock
    from diffusion_editor.evaluation.artifacts import count_unet
    service._models()
    service._editor.edit_global = Mock(side_effect=AssertionError("Unrequested editing"))
    vae = service._backend.vae
    vae.decode = Mock(wraps=vae.decode)
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, None, None, DemoRequest(mode="Global edit", target_prompt="", steps=3,
                            include_edit=False, include_reconstruction=True, refine_reconstruction=False))
    assert budget["unet_calls"] == 6
    assert vae.decode.call_count == 1
    assert result.reconstruction and result.vae is None and result.edited is None
    assert result.report["computed_outputs"] == [DDIM]


def test_source_control_only_uses_local_mask_and_no_target(service, image):
    from unittest.mock import Mock
    from diffusion_editor.evaluation.artifacts import count_unet
    mask = Image.new("L", image.size)
    service._models()
    vae = service._backend.vae
    vae.decode = Mock(wraps=vae.decode)
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, mask, None, DemoRequest(target_prompt="", steps=3, strength=0.5,
                            include_edit=False, include_source_control=True))
    assert budget["unet_calls"] == 5
    assert vae.decode.call_count == 1
    assert result.control and result.edited is None
    assert result.report["target_timesteps"] is None
    np.testing.assert_array_equal(np.asarray(result.control), np.asarray(image))


def test_combined_outputs_decode_one_shared_vae_baseline(service, image):
    from unittest.mock import Mock
    from diffusion_editor.evaluation.artifacts import count_unet
    service._models()
    vae = service._backend.vae
    vae.decode = Mock(wraps=vae.decode)
    with count_unet(service._backend.unet) as budget:
        result = service.run(image, None, None, DemoRequest(mode="Global edit", steps=3, strength=0.5,
            include_vae_reconstruction=True, include_reconstruction=True, include_source_control=True,
            refine_reconstruction=False))
    assert budget["unet_calls"] == 13  # Edit with control: 7; independent plain reconstruction: 6.
    assert vae.decode.call_count == 4  # One baseline, edit, source control and DDIM output.
    assert set(result.report["computed_outputs"]) == {EDIT, VAE, DDIM, CONTROL}
