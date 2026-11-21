"""Opt-in Gradio tab composition and isolated operation routing checks."""
import socket

from PIL import Image
import pytest

from diffusion_editor.demo.ui_state import DDIM, EDIT
from test_generation import generation_service


@pytest.mark.demo
def test_persistent_independent_tabs(generation_service):
    pytest.importorskip("gradio")
    from diffusion_editor.demo.app import create_app
    app = create_app(service=generation_service)
    components = app.config["components"]
    tabs = [c for c in components if c["type"] == "tabitem"]
    assert [c["props"]["label"] for c in tabs] == ["Text to image", "Local edit", "Global edit", "Reconstruction",
                                                   "Img2Img (noise)", "Inpainting (noise)", "Outpainting (noise)"]
    assert all(c["props"]["render_children"] for c in tabs)
    navigation = next(c for c in components if c["type"] == "tabs")
    assert navigation["props"]["selected"] == "local"
    assert sum(c["type"] == "imageeditor" for c in components) == 1
    operations = [c for c in components if c["props"].get("label") == "Operation"]
    assert [c["props"]["value"] for c in operations] == ["Local edit", "Global edit", "Reconstruct source"]
    assert all(c["props"]["visible"] is False for c in operations)
    ids = [c["props"]["elem_id"] for c in components if c["props"].get("elem_id")]
    assert len(ids) == len(set(ids))
    assert {"local-mask-panel", "global-target-prompt", "reconstruction-target-prompt"} <= set(ids)
    assert not generation_service.load_calls


@pytest.mark.demo
def test_each_tab_api_uses_one_shared_cache(generation_service, tmp_path):
    pytest.importorskip("gradio")
    from gradio_client import Client, handle_file
    from diffusion_editor.demo.app import create_app
    image = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "red").save(image)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(service=generation_service)
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True,
                   allowed_paths=[str(generation_service.output_dir)], quiet=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        generated = client.predict("apple", "", 64, 64, 3, 4, 42, api_name="/txt2img")
        global_result = client.predict(image=handle_file(image), operation="Global edit",
            source_prompt="coffee", target_prompt="pencil", num_steps=3,
            chosen_outputs=[EDIT], api_name="/run_global")
        reconstructed = client.predict(image=handle_file(image), operation="Reconstruct source",
            source_prompt="coffee", num_steps=3, refine_source=False,
            chosen_outputs=[DDIM], api_name="/run_reconstruction")
        assert global_result[8]["request"]["mode"] == "Global edit"
        assert global_result[8]["background_metrics"] is None
        assert reconstructed[8]["request"]["mode"] == "Reconstruct source"
        assert reconstructed[8]["computed_outputs"] == [DDIM]
        assert generated[4]["model_first_load"]
        assert not global_result[8]["model_first_load"] and not reconstructed[8]["model_first_load"]
        assert len(generation_service.load_calls) == 1
        assert len({generated[2], global_result[9], reconstructed[9]}) == 3
    finally:
        app.close()
