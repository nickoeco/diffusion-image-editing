"""Task routing, geometry and provenance checks using offline synthetic components."""
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pytest

from diffusion_editor.demo.image_fit import prepare_upload
from diffusion_editor.demo.inputs import editor_canvas
from diffusion_editor.demo.ui_state import CONTROL, DDIM, VAE
from diffusion_editor.demo.workspace_run import execute_task
from diffusion_editor.demo.workspace_state import (boundary_comparison, frozen_settings, mask_overlay, outpaint_preview,
                                                   planned_reverse_steps, result_badge, settings_digest)
from diffusion_editor.editing.canvas import prepare_outpaint_canvas
from test_generation import generation_service


@pytest.fixture
def settings():
    image = Image.new("RGB", (64, 64), (90, 130, 160))
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).rectangle((20, 20, 43, 43), fill=255)
    return dict(image=image, prepared=prepare_upload(image, enabled=False), uploaded_mask=mask,
        paint=editor_canvas(image), scope="Selected region", method="Forward noising", prompt="pencil",
        source_prompt="coffee", steps=3, strength=.85, guidance=4, seed=42, negative="", mask_source="Upload",
        soft=False, feather=0, latent=True, preserve=True, inversion_guidance=1,
        selected_outputs=[DDIM], refine=False, intermediates=False, left=0, right=16, top=0,
        bottom=0, overlap=0, fill="edge", width=64, height=64, fit=False)


def test_settings_association_tracks_paint_and_keeps_zero_distinct(settings):
    digest = settings_digest(settings)
    saved = {"run_id": "immutable-run", "settings_digest": digest}
    assert "matches" in result_badge(saved, digest)
    assert "previous result" in result_badge(saved, settings_digest(settings | {"strength": 0}))
    layer = Image.new("RGBA", (64, 64), (255, 255, 255, 128))
    settings["paint"]["layers"].append(layer)
    assert settings_digest(settings) != digest
    assert saved["settings_digest"] == digest
    assert result_badge(None, digest) == "No completed run yet."


def test_frozen_paint_reference_matches_actual_layers_without_live_exports(settings):
    layer = Image.new("RGBA", settings["image"].size, (255, 255, 255, 128))
    settings["paint"]["layers"].append(layer)
    frozen = frozen_settings(settings["paint"])
    assert settings_digest(settings) == settings_digest(settings | {"paint": frozen})
    layer.putpixel((3, 4), (0, 0, 0, 0))
    assert settings_digest(settings) != settings_digest(settings | {"paint": frozen})


def test_paint_signature_ignores_editor_export_details_not_used_by_mask_resolution(settings):
    empty = settings_digest(settings)
    settings["paint"]["layers"] = [Image.new("RGBA", settings["image"].size)]
    settings["paint"]["composite"] = Image.new("RGBA", settings["image"].size, "white")
    settings["paint"]["background"] = settings["image"].convert("RGB")
    assert settings_digest(settings) == empty
    settings["paint"]["layers"][0].putpixel((2, 3), (10, 20, 30, 128))
    painted = settings_digest(settings)
    settings["paint"]["layers"][0].putpixel((2, 3), (255, 255, 255, 128))
    assert settings_digest(settings) == painted != empty


@pytest.mark.parametrize("strength,floor_count,ceil_count", [(.14, 7, 7), (.58, 29, 29), (.01, 0, 1), (0, 0, 0), (1, 50, 50)])
def test_planned_reverse_count_uses_exact_route_conventions(strength, floor_count, ceil_count):
    assert planned_reverse_steps("Forward noising", 50, strength) == floor_count
    assert planned_reverse_steps("DDIM inversion", 50, strength) == ceil_count


@pytest.mark.parametrize("fill", ["edge", "neutral"])
@pytest.mark.parametrize("overlap", [0, 5])
def test_canvas_preview_matches_sampler_geometry(fill, overlap):
    image = Image.fromarray(np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3))
    preview, mask, note = outpaint_preview(image, 8, 16, 8, 0, overlap, fill)
    actual = prepare_outpaint_canvas(image, left=8, right=16, top=8, overlap=overlap, fill=fill)
    np.testing.assert_allclose(np.asarray(preview) / 255, actual.pixels[0].permute(1, 2, 0).numpy(), atol=1/255)
    np.testing.assert_allclose(np.asarray(mask) / 255, actual.mask[0, 0].numpy(), atol=1/255)
    assert preview.crop((8, 8, 72, 56)).tobytes() == image.tobytes()
    assert str(actual.source_box) in note


@pytest.mark.parametrize("right,overlap", [(0, 0), (7, 0), (2048, 0), (8, 32), (8, -1), (8.5, 0)])
def test_canvas_preview_rejects_invalid_geometry(right, overlap):
    with pytest.raises(ValueError):
        outpaint_preview(Image.new("RGB", (64, 64)), 0, right, 0, 0, overlap, "edge")


def test_boundary_is_fixed_by_requested_mask_not_output(settings):
    before, mask = settings["image"], settings["uploaded_mask"]
    first, note = boundary_comparison(before, Image.new("RGB", before.size, "white"), mask, side=24)
    second, second_note = boundary_comparison(before, Image.new("RGB", before.size, "black"), mask, side=24)
    assert note == second_note and first[0].tobytes() == second[0].tobytes()
    assert first[0].size == (24, 24)
    assert boundary_comparison(before, before, Image.new("L", before.size))[0] is None
    overlay = np.asarray(mask_overlay(before, mask))
    np.testing.assert_array_equal(overlay[np.asarray(mask) == 0], np.asarray(before)[np.asarray(mask) == 0])


@pytest.mark.parametrize("task,scope,expected", [("edit", "Whole image", "img2img"),
    ("edit", "Selected region", "inpaint"), ("extend", "Whole image", "outpaint")])
def test_forward_workspace_never_calls_inversion_and_saves_association(generation_service, settings, task, scope, expected):
    generation_service.run = Mock(side_effect=AssertionError("Forward route must not invert"))
    values = settings | {"scope": scope}
    result = execute_task(generation_service, task, values, lambda *_: None)
    assert result.report["mode"] == expected and not result.report["inversion_executed"]
    assert not generation_service.run.called
    saved = json.loads((Path(result.result_file).parent / "workspace.json").read_text())
    assert saved["run_id"] == result.run_id and saved["settings_digest"] == settings_digest(values)
    assert saved["source_prompt"] == "" and saved["target_prompt"] == "pencil"
    with zipfile.ZipFile(result.archive_file) as archive:
        assert "workspace.json" in archive.namelist()
        assert json.loads(archive.read("report.json"))["request"]["mode"] == expected


def test_draw_and_explicit_fitting_preserve_alignment(generation_service, settings):
    original = Image.new("RGB", (61, 57), (90, 130, 160))
    prepared = prepare_upload(original, enabled=True)
    paint = editor_canvas(prepared.image)
    layer = Image.new("RGBA", prepared.image.size)
    ImageDraw.Draw(layer).rectangle((20, 20, 40, 40), fill=(255, 255, 255, 255))
    paint["layers"].append(layer)
    values = settings | dict(image=prepared.image, prepared=prepared, paint=paint, mask_source="Draw", fit=True)
    result = execute_task(generation_service, "edit", values, lambda *_: None)
    assert result.image.size == (64, 64) and result.requested_mask.getpixel((25, 25)) == 255
    np.testing.assert_array_equal(np.asarray(result.image)[0, 0], np.asarray(prepared.image)[0, 0])
    with zipfile.ZipFile(result.archive_file) as archive:
        assert "workspace_original_upload.png" in archive.namelist()
        manifest = json.loads(archive.read("workspace.json"))
        assert manifest["preprocessing"]["source_size"] == [61, 57]
        assert manifest["ui_preprocessing"]["mask_input"] == "Draw"
    values["paint"] = editor_canvas(Image.new("RGB", prepared.image.size))
    with pytest.raises(ValueError, match="different image"):
        execute_task(generation_service, "edit", values, lambda *_: None)


def test_inversion_and_diagnostic_outputs_are_opt_in(generation_service, settings):
    generation_service.forward_edit = Mock(side_effect=AssertionError("No forward route"))
    edited = execute_task(generation_service, "edit", settings | {"method": "DDIM inversion"}, lambda *_: None)
    assert edited.report["computed_outputs"] == ["Edited image"]
    diagnostic = execute_task(generation_service, "diagnostics", settings | {"selected_outputs": [VAE]}, lambda *_: None)
    assert diagnostic.report["computed_outputs"] == [VAE]
    assert diagnostic.report["reconstruction_timesteps"] is None
    control = execute_task(generation_service, "diagnostics", settings | {"selected_outputs": [CONTROL]}, lambda *_: None)
    assert control.report["computed_outputs"] == [CONTROL]
    assert control.target_prompt == "" and control.image is not None
    assert not generation_service.forward_edit.called and len(generation_service.load_calls) == 1


def test_generation_has_no_before_or_source_prompt(generation_service, settings):
    result = execute_task(generation_service, "generate", settings, lambda *_: None)
    assert result.before is None and result.requested_mask is None and result.source_prompt == ""
    assert result.report["mode"] == "txt2img"


@pytest.mark.demo
def test_workspace_mask_upload_preflight_and_failure_retention(generation_service, tmp_path, settings):
    gr = pytest.importorskip("gradio")
    from diffusion_editor.demo.app import create_app
    app = create_app(service=generation_service)
    editors = {component._id for component in app.blocks.values() if component.__class__.__name__ == "ImageEditor"}
    refreshers = [event for event in app.fns.values() if event.fn and event.fn.__name__ == "refresh"]
    assert all(not editors.intersection(component._id for component in event.inputs) for event in refreshers)
    sync = next(event.fn for event in app.fns.values() if event.fn and event.fn.__name__ == "sync_paint")
    assert sync(settings["paint"], settings["prepared"]) == frozen_settings(settings["paint"])
    assert sync(editor_canvas(Image.new("RGB", (64, 64))), settings["prepared"]) == gr.skip()
    load_mask = next(event.fn for event in app.fns.values() if event.fn and event.fn.__name__ == "load_mask")
    wrong = tmp_path / "wrong.png"
    Image.new("L", (5, 5), 255).save(wrong)
    with pytest.raises(gr.Error, match="not aligned"):
        load_mask(str(wrong), settings["image"], settings["prepared"], False, False)
    valid = tmp_path / "valid.png"
    settings["uploaded_mask"].save(valid)
    mask, source, view, message = load_mask(str(valid), settings["image"], settings["prepared"], False, False)
    assert mask.size == (64, 64) and (source, view) == ("Upload", "Input") and "Mask loaded" in message
    # A failed UI run returns skip updates for images, report, downloads and snapshot.
    execute = next(event for event in app.fns.values() if event.fn and event.fn.__name__ == "execute"
                   and any(getattr(component, "elem_id", None) == "workspace-edit-prompt" for component in event.inputs))
    values = [component.value for component in execute.inputs]
    values[0], values[1], values[2] = settings["prepared"], settings["image"], None
    paint = next(i for i, component in enumerate(execute.inputs)
                 if getattr(component, "elem_id", None) == "workspace-edit-paint")
    values[paint] = editor_canvas(settings["image"])
    # Use Draw explicitly rather than depending on any serialized example mask.
    mask_source = next(i for i, component in enumerate(execute.inputs)
                       if getattr(component, "elem_id", None) == "workspace-edit-mask-source")
    values[mask_source] = "Draw"
    result = execute.fn(*values)
    assert "No painted region" in result[7]
    assert all(update == gr.skip() for i, update in enumerate(result) if i != 7)
