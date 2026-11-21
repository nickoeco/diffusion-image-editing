"""Launch the custom SD editor and the isolated, labelled FLUX reference page."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

from diffusion_editor.config import ProjectConfig, load_config

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def create_app(config: ProjectConfig | None = None, output_dir: str | Path = "outputs/demo",
               *, service: Any = None, flux_service: Any = None,
               flux_python: str | Path | None = None, flux_config: str | Path | None = None) -> Any:
    import gradio as gr
    from PIL import Image
    from diffusion_editor.demo.inputs import editor_canvas, normalize_image
    from diffusion_editor.demo.image_fit import prepare_upload
    from diffusion_editor.demo.service import DemoRequest, DemoService, GenerationRequest, MODES
    from diffusion_editor.demo.ui_state import CONTROL, DDIM, EDIT, VAE, default_outputs, layout_css, output_choices, validate_outputs

    service = service or DemoService(config or ProjectConfig(), output_dir)
    if flux_service is None:
        from diffusion_editor.demo.flux_service import FluxDemoService
        flux_service = FluxDemoService(service, python=flux_python, config_path=flux_config)
    with Image.open(ROOT / "assets/examples/coffee.png") as image:
        coffee = image.convert("RGB")
    with Image.open(ROOT / "assets/examples/coffee_mask.png") as mask:
        coffee_mask = mask.convert("L")
    defaults = DemoRequest()
    initial = prepare_upload(coffee)

    def build_editor(operation: str, api_suffix: str) -> None:
        """Build tab-local controls/state while reusing the shared model service."""
        prefix = api_suffix.lstrip("_") + "-" if api_suffix else ""

        def element_id(name: str) -> str:
            return prefix + name

        def scoped_css(operation: str, chosen: list[str]) -> str:
            return layout_css(operation, chosen).replace("#", "#" + prefix)

        chosen = default_outputs(operation)
        visibility_style = gr.HTML(scoped_css(operation, chosen))
        input_state = gr.State(initial)
        with gr.Row():
            with gr.Column():
                original = gr.Image(value=initial.image, type="pil", image_mode="RGB", format="png",
                                    sources=["upload"], interactive=True, label="Original image", height=320)
                with gr.Row():
                    custom_upload = gr.UploadButton("Upload custom image", file_types=["image"], type="filepath", variant="primary")
                    example = gr.Button("Load coffee example")
                auto_fit = gr.Checkbox(True, label="Automatically resize and pad",
                                      info="Changing this clears masks and results. Disabled: strict size validation.")
                size_info = gr.Textbox(value=initial.fit.summary(), label="Image sizing", interactive=False)
                mode = gr.Radio(list(MODES), value=operation, label="Operation", visible=False)
                selected_outputs = gr.CheckboxGroup(output_choices(operation), value=default_outputs(operation),
                    label="Outputs to calculate", info="Only selected outputs are computed. Selections and results stay in this tab.")
                source = gr.Textbox(value=defaults.source_prompt, label="Source prompt", lines=2)
                target = gr.Textbox(value=defaults.target_prompt, label="Target prompt", lines=2, elem_id=element_id("target-prompt"))
            with gr.Column():
                if operation == "Local edit":
                    with gr.Group(elem_id=element_id("local-mask-panel")):
                        gr.Markdown("Mask settings apply to **Local edit**. Choose Upload or Draw; the other mask is ignored.")
                        mask_source = gr.Radio(["Upload", "Draw"], value="Upload", label="Mask source")
                        uploaded = gr.Image(value=coffee_mask, type="pil", image_mode="L", format="png",
                                            sources=["upload"], interactive=True, label="Mask upload — original or canvas size; white edits", height=200)
                        drawing = gr.ImageEditor(value=editor_canvas(initial.image), type="pil", image_mode="RGBA", format="png",
                                                 sources=(), transforms=(), layers=False, interactive=True,
                                                 brush=gr.Brush(colors=["#ffffff"], color_mode="fixed", default_size=30),
                                                 eraser=gr.Eraser(default_size=30),
                                                 label="Draw on the original (used only when Draw is selected)", height=320)
                        clear_drawing = gr.Button("Clear painted region")
                        soft_mask = gr.Checkbox(False, label="Use soft mask weights")
                        feather = gr.Slider(0, 32, value=0, step=1, label="Mask feathering (pixels)")
                        latent = gr.Checkbox(True, label="Blend source latents outside the edit region")
                        preserve = gr.Checkbox(True, label="Preserve original background pixels")
                else:
                    mask_source, uploaded, drawing = gr.State("Upload"), gr.State(None), gr.State(None)
                    soft_mask, feather = gr.State(False), gr.State(0)
                    latent, preserve = gr.State(True), gr.State(True)
                steps = gr.Slider(1, 200, value=50, step=1, label="Sampling steps")
                with gr.Group(elem_id=element_id("edit-parameters")):
                    strength = gr.Slider(0, 1, value=1, step=0.05, label="Edit strength")
                    guidance = gr.Slider(0, 15, value=7.5, step=0.5, label="Edit guidance")
                seed = gr.Number(42, precision=0, label="Seed",
                                 info="Recorded for reproducibility; deterministic inversion adds no random edit noise.")
        with gr.Accordion("Reconstruction and advanced settings", open=False):
            with gr.Group(elem_id=element_id("reconstruction-settings")):
                refine_reconstruction = gr.Checkbox(True, label="Use residual correction for DDIM reconstruction (slower)")
                gr.Markdown("DDIM reconstruction uses the source prompt independently of editing. "
                            "This correction changes the reconstruction output only.")
            inversion_guidance = gr.Slider(0, 10, value=1, step=0.5, label="Editing source inversion guidance", elem_id=element_id("inversion-guidance"))
            negative = gr.Textbox(label="Negative prompt", value="")
            intermediates = gr.Checkbox(False, label="Show decoded-latent previews", elem_id=element_id("intermediate-settings"))
        run = gr.Button("Run", variant="primary")
        status = gr.Textbox(label="Run status", interactive=False)
        with gr.Row():
            original_out = gr.Image(label="Original", type="pil", format="png", interactive=False)
            vae_out = gr.Image(label="VAE reconstruction", type="pil", format="png", interactive=False, visible=VAE in chosen)
        with gr.Row():
            reconstruction_out = gr.Image(label="Full DDIM source reconstruction", type="pil", format="png", interactive=False, visible=DDIM in chosen)
            edited_out = gr.Image(label="Edited image", type="pil", format="png", interactive=False, visible=EDIT in chosen)
        with gr.Accordion("Control image, mask and intermediate previews", open=False):
            with gr.Row():
                control_out = gr.Image(label="Source-prompt editing control", type="pil", format="png", interactive=False, visible=CONTROL in chosen)
                mask_out = gr.Image(label="Effective edit mask", type="pil", format="png", interactive=False, visible=operation == "Local edit")
            previews = gr.Gallery(label="Optional decoded-latent previews", columns=3, height=260)
        with gr.Row():
            result_download = gr.File(label="Download result PNG", interactive=False)
            archive_download = gr.File(label="Download run archive (images and report)", interactive=False)
        with gr.Accordion("Reproducibility report", open=False):
            report = gr.JSON(label="Parameters, timings and metrics")
        gr.Markdown("Local masks preserve pixels where the effective mask is zero when background compositing is enabled. "
                    "Feathering expands the effective region. Text edits can change geometry; color edits remain unreliable.")

        def execute(image, uploaded_mask, editor_value, operation, source_prompt, target_prompt,
                    num_steps, edit_strength, edit_guidance, random_seed, mask_choice, soft_weights,
                    feather_pixels, blend_latents, preserve_pixels, source_guidance, negative_prompt,
                    chosen_outputs, refine_source, show_previews, fit_enabled, prepared_state, progress=gr.Progress()):
            try:
                validate_outputs(operation, chosen_outputs)
                request = DemoRequest(mode=operation, source_prompt=source_prompt, target_prompt=target_prompt,
                                      steps=_integer(num_steps, "Steps"), strength=edit_strength,
                                      guidance=edit_guidance, seed=_integer(random_seed, "Seed"),
                                      mask_source=mask_choice, soft_mask=soft_weights, feather_radius=feather_pixels,
                                      latent_blending=blend_latents, preserve_background=preserve_pixels,
                                      inversion_guidance=source_guidance, negative_prompt=negative_prompt,
                                      include_reconstruction=DDIM in chosen_outputs,
                                      include_vae_reconstruction=VAE in chosen_outputs,
                                      include_edit=EDIT in chosen_outputs, include_source_control=CONTROL in chosen_outputs,
                                      refine_reconstruction=refine_source, show_intermediates=show_previews, auto_fit=fit_enabled)
                result = service.run(image, uploaded_mask, editor_value, request,
                                     progress=lambda fraction, message: progress(fraction, desc=message), input_state=prepared_state)
                return (result.original, gr.update(value=result.vae, visible=VAE in chosen_outputs),
                        gr.update(value=result.reconstruction, visible=DDIM in chosen_outputs),
                        gr.update(value=result.edited, visible=EDIT in chosen_outputs),
                        gr.update(value=result.control, visible=CONTROL in chosen_outputs),
                        result.mask, result.previews, result.status, result.report, result.result_file, result.archive_file)
            except Exception as exc:
                logger.exception("Demo run failed")
                raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

        outputs = [original_out, vae_out, reconstruction_out, edited_out, control_out, mask_out,
                   previews, status, report, result_download, archive_download]
        inputs = [original, uploaded, drawing, mode, source, target, steps, strength, guidance, seed,
                  mask_source, soft_mask, feather, latent, preserve, inversion_guidance, negative,
                  selected_outputs, refine_reconstruction, intermediates, auto_fit, input_state]
        run.click(execute, inputs, outputs, api_name="run" + api_suffix, concurrency_limit=1, concurrency_id="inference")
        # CSS hides inactive controls while retaining the initialized ImageEditor.
        # Toggling Gradio's visible property unmounts its canvas and can lose tools.

        def cleared_outputs(operation, chosen, message):
            return (None, gr.update(value=None, visible=VAE in chosen),
                    gr.update(value=None, visible=DDIM in chosen), gr.update(value=None, visible=EDIT in chosen),
                    gr.update(value=None, visible=CONTROL in chosen), None, [], message, None, None, None)

        def change_operation(operation):
            chosen = default_outputs(operation)
            return (gr.update(choices=output_choices(operation), value=chosen), scoped_css(operation, chosen),
                    *cleared_outputs(operation, chosen, "Operation changed. Select outputs, then run again."))

        def change_outputs(operation, chosen):
            return (scoped_css(operation, chosen),
                    *cleared_outputs(operation, chosen, "Output selection changed. Run to calculate the selected results."))

        mode.change(change_operation, mode, [selected_outputs, visibility_style] + outputs,
                    queue=False, api_name="change_operation" + api_suffix)
        selected_outputs.change(change_outputs, [mode, selected_outputs], [visibility_style] + outputs,
                                queue=False, api_name="change_outputs" + api_suffix)

        def reset_input(image, operation, chosen, fit_enabled, prepared_state):
            prepared = prepare_upload(image, prepared_state, enabled=fit_enabled) if image is not None else None
            fitted = prepared.image if prepared else None
            return (fitted, editor_canvas(fitted), None, prepared,
                    prepared.fit.summary() if prepared else "Upload an original image first.",
                    *cleared_outputs(operation, chosen,
                    "Original changed. For a local edit, upload or draw an aligned mask, then run again."))

        reset_inputs = [original, mode, selected_outputs, auto_fit, input_state]
        reset_outputs = [original, drawing, uploaded, input_state, size_info] + outputs
        original.upload(reset_input, reset_inputs, reset_outputs, queue=False, api_name=False)
        original.clear(reset_input, reset_inputs, reset_outputs, queue=False, api_name=False)
        auto_fit.input(reset_input, reset_inputs, reset_outputs, queue=False, api_name="fit_image" + api_suffix)

        def upload_image(path, operation, chosen, fit_enabled):
            try:
                if not path:
                    raise ValueError("Choose an image file to upload.")
                with Image.open(path) as value:
                    image = normalize_image(value)
                return reset_input(image, operation, chosen, fit_enabled, None)
            except Exception as exc:
                raise gr.Error(f"Cannot open image: {exc}") from exc

        custom_upload.upload(upload_image, [custom_upload, mode, selected_outputs, auto_fit], reset_outputs,
                             queue=False, api_name="upload_image" + api_suffix)
        if operation == "Local edit":
            clear_drawing.click(lambda image: (editor_canvas(image), "Draw"), original, [drawing, mask_source],
                                queue=False, api_name=False)
        def load_example():
            chosen = default_outputs(operation)
            return (initial.image.copy(), coffee_mask.copy(), editor_canvas(initial.image), defaults.source_prompt,
                    defaults.target_prompt, operation, "Upload", gr.update(choices=output_choices(operation), value=chosen),
                    scoped_css(operation, chosen), True, initial, initial.fit.summary(),
                    *cleared_outputs(operation, chosen, "Coffee example loaded. Ready to run."))
        example.click(load_example, outputs=[original, uploaded, drawing, source, target, mode, mask_source,
                      selected_outputs, visibility_style, auto_fit, input_state, size_info] + outputs, queue=False, api_name=False)

    from diffusion_editor.demo.workspace import WORKSPACE_CSS, build_workspace
    with gr.Blocks(title="Diffusion Image Editor", analytics_enabled=False, css=WORKSPACE_CSS,
                   theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:
        build_workspace(gr, service, coffee, coffee_mask, flux_service=flux_service)
        # Keep the established API adapters and their parameter/output order. They
        # share the same service/lock, but do not appear in task navigation.
        with gr.Tabs(selected="local", elem_id="generation-tabs", visible=False):
            with gr.Tab("Text to image", id="txt2img", render_children=True):
                gr.Markdown("Generate from a prompt without an input image or mask. Uses our custom DDIM sampler and CFG with the cached model.")
                generation_prompt = gr.Textbox(label="Generation prompt", value="a red apple on a wooden table", lines=2)
                generation_negative = gr.Textbox(label="Generation negative prompt", value="")
                with gr.Row():
                    generation_width = gr.Slider(64, 1024, value=512, step=64, label="Generation width")
                    generation_height = gr.Slider(64, 1024, value=512, step=64, label="Generation height")
                with gr.Row():
                    generation_steps = gr.Slider(1, 200, value=50, step=1, label="Generation steps")
                    generation_guidance = gr.Slider(0, 15, value=7.5, step=0.5, label="Generation guidance")
                    generation_seed = gr.Number(42, precision=0, label="Generation seed", info="Controls the initial random noise.")
                generate = gr.Button("Generate image", variant="primary")
                generated_image = gr.Image(label="Generated image", type="pil", format="png", interactive=False)
                generation_status = gr.Textbox(label="Generation status", interactive=False)
                with gr.Row():
                    generated_download = gr.File(label="Download generated PNG", interactive=False)
                    generation_archive = gr.File(label="Download generation archive", interactive=False)
                with gr.Accordion("Generation report", open=False):
                    generation_report = gr.JSON(label="Generation settings and runtime")

                def execute_generation(prompt, negative_prompt, width, height, num_steps, guidance, seed,
                                       progress=gr.Progress()):
                    try:
                        request = GenerationRequest(prompt=prompt, negative_prompt=negative_prompt,
                            width=_integer(width, "Width"), height=_integer(height, "Height"),
                            steps=_integer(num_steps, "Steps"), guidance=guidance, seed=_integer(seed, "Seed"))
                        result = service.generate(request, progress=lambda fraction, message: progress(fraction, desc=message))
                        return result.image, result.status, result.result_file, result.archive_file, result.report
                    except Exception as exc:
                        logger.exception("Text-to-image generation failed")
                        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

                generate.click(execute_generation,
                    [generation_prompt, generation_negative, generation_width, generation_height,
                     generation_steps, generation_guidance, generation_seed],
                    [generated_image, generation_status, generated_download, generation_archive, generation_report],
                    api_name="txt2img", concurrency_limit=1, concurrency_id="inference")
            with gr.Tab("Local edit", id="local", render_children=True):
                build_editor("Local edit", "")
            with gr.Tab("Global edit", id="global", render_children=True):
                build_editor("Global edit", "_global")
            with gr.Tab("Reconstruction", id="reconstruction", render_children=True):
                build_editor("Reconstruct source", "_reconstruction")
            from diffusion_editor.demo.forward_app import build_forward_tab
            for label, forward_mode in (("Img2Img (noise)", "img2img"), ("Inpainting (noise)", "inpaint"),
                                        ("Outpainting (noise)", "outpaint")):
                with gr.Tab(label, id="forward-" + forward_mode, render_children=True):
                    build_forward_tab(gr, service, forward_mode, coffee, coffee_mask)
    return demo.queue(default_concurrency_limit=1, max_size=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--flux-python", type=Path, help="Separate Python runtime with FLUX.2 and bitsandbytes support")
    parser.add_argument("--flux-config", type=Path, default=ROOT / "configs/flux-klein-4b-int8.json")
    args = parser.parse_args()
    from diffusion_editor.utils.logging import configure_logging
    config = load_config(args.config)
    configure_logging(config.log_level)
    demo = create_app(config, args.output_dir, flux_python=args.flux_python, flux_config=args.flux_config)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, show_error=True,
                inbrowser=False, allowed_paths=[str(args.output_dir.resolve())], max_file_size="20mb")


if __name__ == "__main__":
    main()
