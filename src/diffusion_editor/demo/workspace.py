"""Task-first Gradio workspace; stable canvases and a shared inference service."""
from __future__ import annotations

from html import escape
import logging
from typing import Any

from PIL import Image, ImageOps

from diffusion_editor.demo.image_fit import prepare_upload
from diffusion_editor.demo.image_fit import image_digest
from diffusion_editor.demo.inputs import editor_canvas, normalize_image
from diffusion_editor.demo.service import DemoRequest
from diffusion_editor.demo.ui_state import CONTROL, DDIM, VAE
from diffusion_editor.demo.workspace_run import execute_task
from diffusion_editor.demo.workspace_state import (boundary_comparison, frozen_settings, mask_overlay, outpaint_preview,
                                                   planned_reverse_steps, result_badge, settings_digest)

logger = logging.getLogger(__name__)

WORKSPACE_CSS = """
.gradio-container {max-width: 1600px !important;}
#workspace-header {padding: 8px 4px 12px;}
#workspace-header h1 {font-size: 26px; margin-bottom: 4px;}
.workspace-row {align-items: flex-start !important; gap: 24px !important;}
.workspace-sidebar {position: sticky; top: 12px; flex: 0 0 340px !important; min-width: 300px !important;}
.workspace-controls {max-height: calc(100vh - 400px); overflow-y: auto; padding: 4px 8px 8px 2px;}
.workspace-controls > * {flex-shrink: 0 !important;}
.workspace-controls h3 {font-size: 15px !important; margin: 0 !important;}
.workspace-controls .prose {font-size: 12px; line-height: 1.4;}
#workspace-edit-scope label {padding: 5px 7px; font-size: 13px;}
#workspace-edit-scope .wrap {gap: 4px;}
.workspace-run {position: sticky; bottom: 0; z-index: 2;}
.workspace-stage {min-width: 400px !important; flex: 1 1 400px !important;}
.workspace-status textarea {font-size: 12px !important;}
.workspace-summary {font-size: 12px; color: var(--body-text-color-subdued);}
.workspace-provenance {font-size: 13px; overflow-wrap: anywhere;}
.workspace-provenance dt {font-weight: 600; margin-top: 6px;}
.workspace-provenance dd {white-space: pre-wrap; margin: 0;}
@media (max-width: 800px) {
 .workspace-sidebar {position: static; flex: 1 1 100% !important; min-width: 0 !important;}
 .workspace-controls {max-height: none; overflow: visible;}
 .workspace-stage {min-width: 0 !important;}
}
"""


def build_workspace(gr: Any, service: Any, coffee: Image.Image, coffee_mask: Image.Image) -> None:
    gr.Markdown("# Diffusion Image Editor\nChoose a task. Prepare the image, describe your change, then run.", elem_id="workspace-header")
    with gr.Tabs(selected="edit", elem_id="workspace-tabs"):
        for task, label in (("generate", "Generate"), ("edit", "Edit image"),
                            ("extend", "Extend canvas"), ("diagnostics", "Diagnostics")):
            with gr.Tab(label, id=task, render_children=True):
                _build_task(gr, service, task, coffee, coffee_mask)


def _build_task(gr: Any, service: Any, task: str, coffee: Image.Image, coffee_mask: Image.Image) -> None:
    """One independent page per task. Never unmount an initialized paint canvas."""
    defaults = DemoRequest()
    prefix = "workspace-" + task
    fields: dict[str, Any] = {}
    initial = prepare_upload(coffee, enabled=False)
    snapshot = gr.State(None)
    fields["prepared"] = gr.State(initial)
    fields["image"] = gr.Image(value=coffee, type="pil", image_mode="RGB", format="png", visible=False)
    fields["uploaded_mask"] = gr.State(coffee_mask.copy() if task == "edit" else None)
    style = gr.HTML("")

    def size_label(prepared):
        source, canvas = prepared.original.size, prepared.image.size
        return f"Source {source[0]} × {source[1]} → canvas {canvas[0]} × {canvas[1]} px."

    def control(name: str, component: Any) -> Any:
        fields[name] = component
        return component

    with gr.Row(elem_classes="workspace-row"):
        with gr.Column(scale=0, min_width=300, elem_classes="workspace-sidebar"):
            with gr.Column(elem_classes="workspace-controls"):
                upload = example = mask_upload = clear = None
                if task != "generate":
                    gr.Markdown("### 1 · Input image")
                    with gr.Row():
                        upload = gr.UploadButton("Upload image", type="filepath", file_types=["image"], size="sm",
                                                 min_width=120, elem_id=prefix + "-upload")
                        example = gr.Button("Coffee example", size="sm", min_width=120)
                    sizing = gr.Markdown(size_label(initial), elem_classes="workspace-summary")
                if task == "edit":
                    control("scope", gr.Radio(["Whole image", "Selected region"], value="Selected region",
                                               label="Edit area", container=False, elem_id=prefix + "-scope"))
                    with gr.Group(elem_id=prefix + "-mask-controls"):
                        control("mask_source", gr.Radio(["Upload", "Draw"], value="Upload", label="Active mask", container=False,
                                                        elem_id=prefix + "-mask-source"))
                        with gr.Group(elem_id=prefix + "-upload-mask-controls"):
                            mask_upload = gr.UploadButton("Upload grayscale mask", type="filepath", file_types=["image"],
                                                          elem_id=prefix + "-mask-upload")
                        with gr.Group(elem_id=prefix + "-draw-mask-controls"):
                            clear = gr.Button("Clear painted region", size="sm", elem_id=prefix + "-clear-paint")
                        gr.Markdown("White / cyan edits · black preserves.",
                                    elem_classes="workspace-summary")
                if task == "extend":
                    gr.Markdown("### 2 · Expand canvas")
                    with gr.Row():
                        control("left", gr.Number(0, precision=0, label="Left (px)", min_width=90))
                        control("right", gr.Number(128, precision=0, label="Right (px)", min_width=90,
                                                   elem_id=prefix + "-right"))
                    with gr.Row():
                        control("top", gr.Number(0, precision=0, label="Top (px)", min_width=90))
                        control("bottom", gr.Number(0, precision=0, label="Bottom (px)", min_width=90))
                    control("overlap", gr.Number(0, precision=0, label="Editable overlap inside source (px)",
                                                 elem_id=prefix + "-overlap"))
                if task == "diagnostics":
                    gr.Markdown("### 2 · Select calculations")
                    control("selected_outputs", gr.CheckboxGroup([VAE, DDIM, CONTROL], value=[DDIM], label="Outputs to calculate",
                        info="Only selected outputs are computed. Controls replay the source prompt through the editing route."))
                    control("source_prompt", gr.Textbox(value=defaults.source_prompt, label="Source prompt", lines=3))
                else:
                    gr.Markdown("### 1 · Describe the result" if task == "generate" else "### 2 · Describe the result" if task == "edit" else "### 3 · Describe the extension")
                    prompt = "a red apple on a wooden table" if task == "generate" else defaults.target_prompt
                    if task == "extend":
                        prompt = "a photo of a cup of coffee on a wooden table, wider view, continuous wooden tabletop"
                    control("prompt", gr.Textbox(value=prompt, label="Prompt" if task == "generate" else "Target prompt",
                                                 lines=3, elem_id=prefix + "-prompt"))
                if task in ("edit", "extend"):
                    control("strength", gr.Slider(0, 1, value=1 if task == "extend" else .85, step=.05, label="Edit strength"))
                if task == "generate":
                    with gr.Row():
                        control("width", gr.Slider(64, 1024, value=512, step=64, label="Width", min_width=100))
                        control("height", gr.Slider(64, 1024, value=512, step=64, label="Height", min_width=100))
                with gr.Accordion("Advanced settings", open=False, elem_id=prefix + "-advanced"):
                    if task == "edit":
                        control("method", gr.Radio(["Forward noising", "DDIM inversion"], value="Forward noising", label="Editing method",
                                                   elem_id=prefix + "-method"))
                        with gr.Group(elem_id=prefix + "-inversion"):
                            control("source_prompt", gr.Textbox(value=defaults.source_prompt, label="Source prompt", lines=3))
                            control("inversion_guidance", gr.Slider(0, 10, value=1, step=.5, label="Source inversion guidance"))
                            control("intermediates", gr.Checkbox(False, label="Decode intermediate previews (extra work)"))
                    control("steps", gr.Slider(1, 200, value=50, step=1, label="Sampling steps", elem_id=prefix + "-steps"))
                    control("guidance", gr.Slider(0, 20, value=7.5, step=.5, label="Guidance (CFG)"))
                    control("seed", gr.Number(42, precision=0, label="Seed"))
                    control("negative", gr.Textbox(value="", label="Negative prompt", lines=2))
                    if task != "generate":
                        control("fit", gr.Checkbox(False, label="Explicitly fit image to model (shrink / pad)",
                            info="Shows the new dimensions and clears both masks. Does not crop or upscale content."))
                    if task in ("edit", "extend"):
                        with gr.Group(elem_id=prefix + "-preservation"):
                            if task == "edit":
                                control("soft", gr.Checkbox(False, label="Use soft mask weights"))
                            control("feather", gr.Slider(0, 32, value=0, step=1, label="Mask feather sigma (px)"))
                            control("latent", gr.Checkbox(True, label="Guide unedited regions",
                                info="Blend source latents at matching updated diffusion levels."))
                            control("preserve", gr.Checkbox(True, label="Keep original pixels outside mask",
                                info="Independent final pixel compositing. Feathering and overlap allow boundary changes."))
                    if task == "extend":
                        control("fill", gr.Radio(["edge", "neutral"], value="edge", label="Canvas initialization"))
                    if task == "diagnostics":
                        control("inversion_guidance", gr.Slider(0, 10, value=1, step=.5, label="Source inversion guidance"))
                        control("refine", gr.Checkbox(True, label="Residual correction for DDIM reconstruction (slower)"))
                        control("intermediates", gr.Checkbox(False, label="Decode intermediate previews (extra work)"))
                summary = gr.Markdown("", elem_classes="workspace-summary")
            run = gr.Button("Generate image" if task == "generate" else "Run diagnostics" if task == "diagnostics" else "Run edit",
                            variant="primary", elem_id=prefix + "-run", elem_classes="workspace-run")
            status = gr.Textbox(value="Ready. No model is loaded until you run.", label="Run status", interactive=False,
                                lines=2, max_lines=4, elem_classes="workspace-status", elem_id=prefix + "-status")
        with gr.Column(elem_classes="workspace-stage"):
            badge = gr.Markdown("No completed run yet.", elem_id=prefix + "-badge")
            view = gr.Radio(["Result"] if task == "generate" else ["Input", "Result", "Compare", "Boundary"], value="Result" if task == "generate" else "Input",
                            label="View", container=False, elem_id=prefix + "-view")
            with gr.Group(elem_id=prefix + "-input-view"):
                initial_preview = mask_overlay(coffee, coffee_mask) if task == "edit" else coffee
                if task == "extend":
                    canvas, weights, _ = outpaint_preview(coffee, 0, 128, 0, 0, 0, "edge")
                    initial_preview = mask_overlay(canvas, weights)
                input_preview = gr.Image(value=initial_preview,
                    type="pil", format="png", height=490, interactive=False, label="Input / requested mask preview",
                    elem_id=prefix + "-input-preview")
                if task == "edit":
                    with gr.Group(elem_id=prefix + "-paint-view"):
                        control("paint", gr.ImageEditor(value=editor_canvas(coffee), type="pil", image_mode="RGBA",
                            format="png", sources=(), transforms=(), layers=False, height=490, interactive=True,
                            brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed", default_size=30),
                            eraser=gr.Eraser(default_size=30), label="Paint the edit region", elem_id=prefix + "-paint"))
                preview_note = gr.Markdown("", elem_classes="workspace-summary")
            with gr.Group(elem_id=prefix + "-result-view"):
                result_image = gr.Image(type="pil", format="png", height=490, interactive=False, label="Result",
                                        elem_id=prefix + "-result")
            with gr.Group(elem_id=prefix + "-compare-view"):
                comparison = gr.ImageSlider(type="pil", format="png", height=490, label="Input ← drag → Result",
                                             interactive=False)
            with gr.Group(elem_id=prefix + "-boundary-view"):
                boundary = gr.ImageSlider(type="pil", format="png", height=490, label="Boundary: input ← drag → result",
                                           interactive=False)
                boundary_note = gr.Markdown("Run a masked edit to inspect a fixed requested-mask boundary crop.")
            with gr.Row():
                png = gr.File(label="Result PNG", interactive=False)
                archive = gr.File(label="Run archive · images / JSON" if task == "generate" else "Run archive · images / JSON / CSV", interactive=False)
            with gr.Accordion("Run details and diagnostics", open=False, elem_id=prefix + "-details"):
                provenance = gr.HTML("Run settings and literal prompts appear here after a successful run.")
                requested_out = gr.Image(type="pil", format="png", label="Requested mask · before feathering", interactive=False, height=200, visible=False)
                effective_out = gr.Image(type="pil", format="png", label="Effective mask · after preprocessing", interactive=False, height=200, visible=False)
                extras = gr.Gallery(label="Selected diagnostics / raw decode", columns=2, height=280, visible=False)
                report = gr.JSON(label="Immutable run report")
                gr.Markdown("Pixel metrics describe preservation. Judge semantic success, unrelated content and boundary quality separately.")

    constants = dict(scope="Whole image", method="Forward noising", prompt="", source_prompt="", steps=50,
        strength=1., guidance=7.5, seed=42, negative="", mask_source="Upload", soft=False, feather=0,
        latent=True, preserve=True, inversion_guidance=1., selected_outputs=[DDIM], refine=True,
        intermediates=False, left=0, right=0, top=0, bottom=0, overlap=0, fill="edge", width=512,
        height=512, fit=False, paint=None)
    for name, value in constants.items():
        if name not in fields:
            fields[name] = gr.State(value)
    names = list(fields)
    inputs = list(fields.values())
    paint_reference = gr.State(frozen_settings(editor_canvas(coffee)) if task == "edit" else None)

    def unpack(values):
        return dict(zip(names, values))

    def page_style(v, selected_view):
        hidden = []
        for key, label in (("input", "Input"), ("result", "Result"), ("compare", "Compare"), ("boundary", "Boundary")):
            if selected_view != label:
                hidden.append(f"#{prefix}-{key}-view")
        if task == "edit":
            local = v["scope"] == "Selected region"
            drawing = local and v["mask_source"] == "Draw"
            hidden.append(f"#{prefix}-input-preview" if drawing else f"#{prefix}-paint-view")
            if not local:
                hidden.extend((f"#{prefix}-mask-controls", f"#{prefix}-preservation"))
            hidden.append(f"#{prefix}-upload-mask-controls" if drawing else f"#{prefix}-draw-mask-controls")
            if v["method"] != "DDIM inversion":
                hidden.append(f"#{prefix}-inversion")
        return "<style>" + ",".join(hidden) + "{display:none !important;}</style>" if hidden else ""

    def refresh(*args):
        v, saved, selected_view = unpack(args[:-2]), args[-2], args[-1]
        try:
            digest = settings_digest(v)
            old_result = result_badge(saved, digest)
            if task == "generate":
                preview, note = None, "Generation starts from seeded noise; no source image."
            else:
                prepared = prepare_upload(v["image"], v["prepared"], enabled=v["fit"])
                preview, note = prepared.image, prepared.fit.summary()
                if task == "extend":
                    canvas, mask, note = outpaint_preview(prepared.image, v["left"], v["right"], v["top"], v["bottom"], v["overlap"], v["fill"])
                    preview = mask_overlay(canvas, mask)
                elif task == "edit" and v["scope"] == "Selected region" and v["mask_source"] == "Upload":
                    if v["uploaded_mask"] is not None:
                        mask, _ = prepared.fit.mask(v["uploaded_mask"], binary=not v["soft"])
                        preview = mask_overlay(preview, mask)
                    note += " Cyan shows the active uploaded mask BEFORE feathering. Paint layers are ignored."
                elif task == "edit" and v["scope"] == "Selected region":
                    note += " Draw is active. Only paint-layer alpha is used; the uploaded mask is ignored."
            method = "Text to image" if task == "generate" else "Reconstruction" if task == "diagnostics" else v["method"] if task == "edit" else "Forward noising"
            settings = f"{method} · {v['steps']} steps · CFG {v['guidance']} · seed {v['seed']}"
            if task in ("edit", "extend"):
                count = planned_reverse_steps(method, v["steps"], v["strength"])
                settings += f" · {count} reverse updates (planned)"
            return page_style(v, selected_view), settings, old_result, preview, note
        except Exception as exc:
            return page_style(v, selected_view), "Check image geometry / settings before running.", result_badge(saved, "invalid"), None, f"Preview unavailable: {exc}"

    # Ordinary control changes must not export the ImageEditor. Concurrent editor
    # exports/reset can race Gradio's temporary PNG writes. A frozen reference is
    # equivalent for hashing, and user paint input updates it separately.
    refresh_inputs = [paint_reference if name == "paint" else fields[name] for name in names] + [snapshot, view]
    refresh_outputs = [style, summary, badge, input_preview, preview_note]
    # Render CSS at construction too: avoid flashing all large images before load.
    initial_values = {name: component.value for name, component in fields.items()}
    initial_values.update(image=coffee, prepared=initial, uploaded_mask=coffee_mask if task == "edit" else None,
                          paint=editor_canvas(coffee) if task == "edit" else None)
    initial_refresh = refresh(*[initial_values[name] for name in names], None, view.value)
    style.value, summary.value, badge.value, _, preview_note.value = initial_refresh
    for component in fields.values():
        if hasattr(component, "input") and component.__class__.__name__ not in ("State", "ImageEditor", "Image"):
            component.input(refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False, show_progress="hidden")
    view.input(refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False, show_progress="hidden")

    def reset_source(path, fit_enabled, is_example=False):
        try:
            if is_example:
                image = coffee.copy()
            else:
                if not path:
                    raise ValueError("Choose an image file.")
                with Image.open(path) as opened:
                    image = normalize_image(opened)
            prepared = prepare_upload(image, enabled=fit_enabled)
            mask = coffee_mask.copy() if is_example and task == "edit" else None
            paint = editor_canvas(prepared.image) if task == "edit" else None
            return prepared.image, prepared, mask, paint, size_label(prepared), "Input prepared. Masks reset; previous results retain their run association.", "Input", frozen_settings(paint)
        except Exception as exc:
            raise gr.Error(f"Cannot prepare image: {exc}") from exc

    if task != "generate":
        reset_outputs = [fields["image"], fields["prepared"], fields["uploaded_mask"], fields["paint"], sizing, status, view, paint_reference]
        upload.upload(reset_source, [upload, fields["fit"]], reset_outputs, queue=False, api_name=False).then(
            refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)
        example.click(lambda enabled: reset_source(None, enabled, True), fields["fit"], reset_outputs,
                      queue=False, api_name=False).then(refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)

        def refit(image, prepared, enabled):
            fitted = prepare_upload(image, prepared, enabled=enabled)
            paint = editor_canvas(fitted.image) if task == "edit" else None
            return fitted.image, fitted, None, paint, size_label(fitted), "Image preparation changed. Both masks were cleared; upload or draw an aligned mask.", "Input", frozen_settings(paint)
        fields["fit"].input(refit, [fields["image"], fields["prepared"], fields["fit"]], reset_outputs,
                            queue=False, api_name=False).then(refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)
    if task == "edit":
        def load_mask(path, image, prepared, enabled, soft):
            try:
                with Image.open(path) as opened:
                    mask = ImageOps.exif_transpose(opened).copy()
                if mask.mode not in ("L", "1"):
                    raise ValueError("Use a grayscale mask: white edits, black preserves.")
                current = prepare_upload(image, prepared, enabled=enabled)
                current.fit.mask(mask, binary=not soft)
                return mask, "Upload", "Input", "Mask loaded. White edits; black preserves. Previous results keep their run association."
            except Exception as exc:
                raise gr.Error(f"Cannot load mask: {exc}") from exc
        mask_upload.upload(load_mask, [mask_upload, fields["image"], fields["prepared"], fields["fit"], fields["soft"]],
                           [fields["uploaded_mask"], fields["mask_source"], view, status],
                           queue=False, api_name=False).then(refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)
        clear.click(lambda image: (editor_canvas(image), "Draw", "Input", frozen_settings(editor_canvas(image))), fields["image"],
                    [fields["paint"], fields["mask_source"], view, paint_reference], queue=False, api_name=False).then(
                        refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)
        def sync_paint(paint, prepared):
            # Gradio 5.49 can suppress .input after a programmatic editor output;
            # .change remains reliable. Discard late exports from a reset canvas.
            background = paint.get("background") if isinstance(paint, dict) else None
            if not isinstance(background, Image.Image) or image_digest(background.convert("RGB")) != prepared.canvas_digest:
                return gr.skip()
            return frozen_settings(paint)

        fields["paint"].change(sync_paint, [fields["paint"], fields["prepared"]], paint_reference, queue=False,
                              api_name=False, show_progress="hidden").then(
                                  refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False,
                                  show_progress="hidden")

    run_outputs = [result_image, comparison, boundary, boundary_note, requested_out, effective_out, extras,
                   status, report, png, archive, provenance, snapshot, view]

    def execute(*args, progress=gr.Progress()):
        v = unpack(args)
        try:
            digest = settings_digest(v)
            result = execute_task(service, task, v, lambda fraction, message: progress(fraction, desc=message))
            crop, crop_note = boundary_comparison(result.before, result.image, result.requested_mask) if result.before is not None else (None, "No spatial mask boundary for text generation.")
            saved = {"run_id": result.run_id, "settings_digest": digest}
            prompts = (("Source prompt", result.source_prompt), ("Target / generation prompt", result.target_prompt))
            details = f'<dl class="workspace-provenance"><dt>Run ID</dt><dd>{escape(result.run_id)}</dd>'
            details += "".join(f"<dt>{label}</dt><dd>{escape(text) if text else '(not used)'}</dd>" for label, text in prompts)
            details += "<dt>Executed settings</dt><dd>" + escape(str(result.report.get("request", {}))) + "</dd></dl>"
            return (result.image, (result.before, result.image) if result.before is not None else None,
                    crop, crop_note, gr.update(value=result.requested_mask, visible=result.requested_mask is not None),
                    gr.update(value=result.effective_mask, visible=result.effective_mask is not None),
                    gr.update(value=result.diagnostics, visible=bool(result.diagnostics)),
                    result.status, result.report, result.result_file, result.archive_file, details, saved, "Result")
        except Exception as exc:
            logger.exception("Workspace %s failed", task)
            values = [gr.skip() for _ in run_outputs]
            values[7] = f"Failed: {type(exc).__name__}: {str(exc).rstrip('.')}. Existing result, if any, was retained."
            return tuple(values)

    run.click(execute, inputs, run_outputs, api_name=False, concurrency_limit=1, concurrency_id="inference").then(
        refresh, refresh_inputs, refresh_outputs, queue=False, api_name=False)
