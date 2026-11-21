"""Offline analytic noising, updated-level masks, canvas coordinates and routing."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import zipfile
import socket

import numpy as np
from PIL import Image
import pytest
import torch

from diffusion_editor.demo.forward_service import ForwardRequest
from diffusion_editor.diffusion.noising import forward_noise
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.editing.canvas import prepare_outpaint_canvas
from diffusion_editor.editing.strength import select_forward_start
from diffusion_editor.pipelines.forward_editing import ForwardEditingPipeline
from test_generation import fake_backend, generation_service


@pytest.fixture
def image():
    return torch.linspace(0, 1, 3 * 16 * 24).reshape(1, 3, 16, 24)


@pytest.mark.parametrize("strength,count", [(0, 0), (0.01, 0), (0.14, 7), (0.25, 12), (0.5, 25), (1, 50)])
def test_forward_floor_strength_actual_levels(strength, count):
    schedule = DDIMSchedule(torch.full((1000,), 0.01), tuple(range(981, 0, -20)))
    result = select_forward_start(schedule, strength)
    assert result.reverse_steps == count and result.start_timestep == schedule.inversion_timesteps[count]


def test_forward_equation_and_invalid_contracts():
    clean, noise = torch.ones(1, 2, 3, 4).half(), torch.full((1, 2, 3, 4), 2.0).half()
    actual = forward_noise(clean, noise, torch.tensor(0.25))
    torch.testing.assert_close(actual, 0.5 * clean.float() + np.sqrt(0.75) * noise.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(forward_noise(clean, noise, torch.tensor(1)), clean.float(), atol=0, rtol=0)
    for alpha in (0, 1.1, float("nan"), [0.5]):
        with pytest.raises(ValueError):
            forward_noise(clean, noise, torch.tensor(alpha))
    with pytest.raises(ValueError):
        forward_noise(clean, noise[..., :2], torch.tensor(0.5))
    with pytest.raises(ValueError):
        forward_noise(clean, noise * float("nan"), torch.tensor(0.5))


@pytest.mark.parametrize("guidance", [0, 1, 5])
def test_analytic_img2img_and_rng_isolation(image, guidance, monkeypatch):
    from diffusion_editor.diffusion.inversion import DDIMInverter
    monkeypatch.setattr(DDIMInverter, "invert", Mock(side_effect=AssertionError("No inversion")))
    pipeline = ForwardEditingPipeline(fake_backend())
    before = torch.random.get_rng_state().clone()
    result = pipeline.img2img(image, "coffee", num_steps=3, strength=1, guidance_scale=guidance, seed=17, return_trajectory=True)
    assert torch.equal(before, torch.random.get_rng_state())
    clean = pipeline.vae.encode_image(image)
    alpha = pipeline.backend.scheduler.alphas_cumprod[6]
    initial = alpha.sqrt() * clean + (1 - alpha).sqrt() * result.noise
    torch.testing.assert_close(result.sampling.trajectory[6], initial)
    expected = (initial - (1 - alpha).sqrt() * (0.1 * guidance)) / alpha.sqrt()
    torch.testing.assert_close(result.sampling.final_latent, expected, atol=1e-6, rtol=1e-6)
    repeat = pipeline.img2img(image, "coffee", num_steps=3, strength=1, guidance_scale=guidance, seed=17)
    assert torch.equal(result.noise, repeat.noise) and torch.equal(result.edited_image, repeat.edited_image)
    changed = pipeline.img2img(image, "coffee", num_steps=3, seed=18)
    assert not torch.equal(changed.noise, result.noise)


def test_zero_and_tiny_strength_do_not_call_unet(image):
    backend = fake_backend()
    backend.unet.forward = Mock(side_effect=AssertionError("No reverse work"))
    pipeline = ForwardEditingPipeline(backend)
    for strength in (0, 0.1):
        result = pipeline.img2img(image, "coffee", num_steps=3, strength=strength)
        assert result.sampling.timesteps == (-1,)
        assert torch.equal(result.edited_image, result.vae_reconstruction)


def test_inpaint_zero_full_partial_and_updated_ids(image):
    pipeline = ForwardEditingPipeline(fake_backend())
    mask = torch.zeros(16, 24)
    mask[4:12, 8:16] = 1
    result = pipeline.inpaint(image, mask, "pencil", num_steps=3, strength=1, return_trajectory=True)
    clean = pipeline.vae.encode_image(image)
    schedule = DDIMSchedule.from_scheduler(pipeline.backend.scheduler, 3)
    zero = result.latent_mask.expand_as(clean) == 0
    for timestep, state in result.sampling.trajectory.items():
        known = forward_noise(clean, result.noise, schedule.alpha(timestep))
        torch.testing.assert_close(state[zero], known[zero], atol=0, rtol=0)
    preserved = (mask == 0).reshape(1, 1, 16, 24).expand_as(image)
    assert torch.equal(result.edited_image[preserved], image[preserved])
    empty = pipeline.inpaint(image, mask * 0, "pencil", num_steps=3)
    assert torch.equal(empty.edited_image, image)
    torch.testing.assert_close(empty.uncomposited_image, empty.vae_reconstruction, atol=0, rtol=0)
    full = pipeline.inpaint(image, mask * 0 + 1, "pencil", num_steps=3)
    global_edit = pipeline.img2img(image, "pencil", num_steps=3, strength=0.85)
    assert torch.equal(full.edited_image, global_edit.edited_image)
    disabled = pipeline.inpaint(image, mask, "pencil", num_steps=3, latent_blending=False, preserve_background=False)
    assert torch.equal(disabled.edited_image, global_edit.edited_image)
    soft = pipeline.inpaint(image.repeat(2, 1, 1, 1), mask * 0.6, ["one", "two"],
                            binary_mask=False, num_steps=3, feather_radius=1)
    assert soft.pixel_mask.shape == (2, 1, 16, 24) and torch.isfinite(soft.edited_image).all()


def test_outpaint_canvas_offsets_overlap_and_exact_retained_source(image):
    prepared = prepare_outpaint_canvas(image, left=4, right=8, top=2, bottom=6)
    assert prepared.pixels.shape == (1, 3, 24, 36) and prepared.source_box == (4, 2, 28, 18)
    assert torch.equal(prepared.pixels[:, :, 2:18, 4:28], image)
    assert not prepared.mask[:, :, 2:18, 4:28].any()
    assert prepared.mask[0, 0, 0, 0] == 1
    ramp = prepare_outpaint_canvas(image, right=8, overlap=3, fill="neutral")
    assert not ramp.mask[:, :, :, :21].any() and ramp.mask[0, 0, 8, 23] == 1
    assert ramp.mask[0, 0, 8, 22] == pytest.approx(2 / 3)
    result = ForwardEditingPipeline(fake_backend()).outpaint(image, "wooden table", left=4, right=8, top=2, bottom=6, num_steps=3)
    assert torch.equal(result.edited_image[:, :, 2:18, 4:28], image)
    assert result.mode == "outpaint" and result.sampling.timesteps[-1] == -1
    for kwargs in (dict(), dict(right=-1), dict(right=8, overlap=8), dict(right=8, fill="bad")):
        with pytest.raises(ValueError):
            prepare_outpaint_canvas(image, **kwargs)


@pytest.mark.parametrize("key,value", [("strength", float("nan")), ("mode", "bad"), ("seed", True),
    ("steps", 0), ("prompt", ""), ("left", -1), ("soft_mask", 1), ("feather_radius", -1)])
def test_invalid_forward_request(key, value):
    with pytest.raises(ValueError):
        ForwardRequest(**({"mode": "img2img", "prompt": "coffee"} | {key: value}))


def test_forward_service_independent_routes_cache_artifacts_and_preflight(generation_service):
    service = generation_service
    image = Image.new("RGB", (64, 64), "red")
    with pytest.raises(ValueError, match="mask"):
        service.forward_edit(image, None, None, ForwardRequest("inpaint", "coffee", steps=3))
    with pytest.raises(ValueError, match="1024"):
        service.forward_edit(image, None, None, ForwardRequest("outpaint", "coffee", right=1024))
    assert not service.load_calls
    service._models()
    service._editor.edit_global = service._editor.edit_local = Mock(side_effect=AssertionError("No inversion route"))
    request = ForwardRequest("img2img", "coffee", steps=3)
    global_edit = service.forward_edit(image, None, None, request)
    inpaint = service.forward_edit(image, Image.new("L", image.size), None, replace(request, mode="inpaint"))
    outpaint = service.forward_edit(image, None, None, replace(request, mode="outpaint", right=16))
    assert len(service.load_calls) == 1 and outpaint.edited.size == (80, 64)
    assert not global_edit.report["inversion_executed"]
    np.testing.assert_array_equal(np.asarray(outpaint.edited)[:, :64], np.asarray(image))
    assert inpaint.report["background_metrics"]["final"]["background_mae"] == 0
    assert Path(inpaint.result_file).parent != Path(outpaint.result_file).parent
    with zipfile.ZipFile(outpaint.archive_file) as archive:
        assert {"original.png", "canvas.png", "edited.png", "raw.png", "requested_mask.png", "noise.pt", "metrics.csv", "report.json"} <= set(archive.namelist())


@pytest.mark.demo
def test_forward_http_modes_draw_and_downloads(generation_service, tmp_path):
    pytest.importorskip("gradio")
    from gradio_client import Client, handle_file
    from diffusion_editor.demo.app import create_app
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (64, 64), "red").save(image)
    Image.new("L", (64, 64)).save(mask)
    layer = tmp_path / "paint.png"
    painting = Image.new("RGBA", (64, 64))
    painting.paste((255, 255, 255, 255), (16, 16, 48, 48))
    painting.save(layer)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(service=generation_service)
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True,
                   allowed_paths=[str(generation_service.output_dir)], quiet=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        for mode in ("img2img", "inpaint", "outpaint"):
            args = dict(image=handle_file(image), target_prompt="coffee", num_steps=3)
            if mode == "inpaint":
                args["mask_image"] = handle_file(mask)
            if mode == "outpaint":
                args["margin_right"] = 16
            result = client.predict(**args, api_name="/forward_" + mode)
            assert result[7]["mode"] == mode and not result[7]["inversion_executed"]
            assert zipfile.is_zipfile(result[6]) and Path(result[5]).is_file()
        drawn = client.predict(image=handle_file(image), target_prompt="coffee", num_steps=3, mask_choice="Draw",
            paint={"background": handle_file(image), "layers": [handle_file(layer)], "composite": handle_file(image)},
            api_name="/forward_inpaint")
        assert drawn[7]["request"]["mask_source"] == "Draw" and len(generation_service.load_calls) == 1
    finally:
        app.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_real_forward_masked_paths(tmp_path):
    from diffusion_editor.config import ProjectConfig
    from diffusion_editor.demo.service import DemoService
    image = Image.open("assets/examples/coffee.png").convert("RGB").resize((64, 64))
    mask = Image.new("L", image.size)
    mask.paste(255, (16, 16, 48, 48))
    service = DemoService(ProjectConfig(), tmp_path)
    for mode in ("img2img", "inpaint", "outpaint"):
        result = service.forward_edit(image, mask if mode == "inpaint" else None, None,
                                     ForwardRequest(mode, "a photo of coffee on a wooden table", steps=4, strength=1, right=16))
        assert result.report["sampling_timesteps"][-1] == -1
        assert not result.report["inversion_executed"] and Path(result.archive_file).is_file()
        if mode != "img2img":
            assert result.report["background_metrics"]["final"]["background_mae"] == 0
