"""Persistent Gradio page for the isolated, labelled FLUX reference backend."""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from diffusion_editor.demo.flux_service import FluxRequest, request_digest
from diffusion_editor.demo.workspace_state import integer
from diffusion_editor.demo.image_fit import ImageFit
from diffusion_editor.demo.inputs import normalize_image

logger = logging.getLogger(__name__)


def build_flux_tab(gr: Any, service: Any, coffee: Image.Image) -> None:
    gr.Markdown("# FLUX · Klein 4B\nGenerate an image or describe a change to an uploaded image. Runs locally with four sampling steps.")
    saved = gr.State(None)
    with gr.Row(elem_classes="workspace-row"):
        with gr.Column(scale=0, min_width=300, elem_classes="workspace-sidebar"):
            with gr.Column(elem_classes="workspace-controls"):
                operation = gr.Radio(["Generate", "Edit image"], value="Generate", label="Task", elem_id="flux-operation")
                with gr.Group(visible=False) as reference_group:
                    original = gr.Image(type="pil", image_mode="RGB", format="png", sources=["upload"],
                                        label="Reference image", height=200, elem_id="flux-reference")
                    fit_reference = gr.Checkbox(False, label="Fit reference to model (shrink / pad)", elem_id="flux-fit")
                    size_info = gr.Markdown("Upload a reference image. Fitting is off by default.", elem_id="flux-sizing")
                    reference_preview = gr.Image(type="pil", format="png", height=200, interactive=False,
                                                 label="Reference sent to FLUX", elem_id="flux-prepared-reference")
                    example = gr.Button("Load coffee example (fit to model)", size="sm", elem_id="flux-example")
                    gr.Markdown("Describe the change in the instruction. This edits the whole image; it does not use a spatial mask.")
                prompt = gr.Textbox(label="Prompt / editing instruction", lines=4, elem_id="flux-prompt",
                                   value="A photograph of a red ceramic mug on a wooden table, soft daylight, a plain neutral background.")
                with gr.Row():
                    width = gr.Slider(64, 1024, value=512, step=16, label="Output width", min_width=100)
                    height = gr.Slider(64, 1024, value=512, step=16, label="Output height", min_width=100)
                seed = gr.Number(42, precision=0, label="Seed", elem_id="flux-seed")
                gr.Markdown("Distilled Klein 4B · protected INT8 · 4 steps · guidance 1", elem_classes="workspace-summary")
            run = gr.Button("Run FLUX", variant="primary", elem_id="flux-run")
            status = gr.Textbox(value="Ready. The FLUX runtime starts only when you run.", label="FLUX status",
                                interactive=False, lines=3, elem_id="flux-status")
        with gr.Column(elem_classes="workspace-stage"):
            badge = gr.Markdown("No completed FLUX run yet.", elem_id="flux-badge")
            result_image = gr.Image(label="FLUX result", type="pil", format="png", interactive=False,
                                    height=490, elem_id="flux-result")
            with gr.Row():
                png = gr.File(label="Result PNG", interactive=False)
                report_file = gr.File(label="Run JSON", interactive=False)
                archive = gr.File(label="Run archive", interactive=False)
            with gr.Accordion("Run details", open=False):
                gr.Markdown("FLUX reference backend. The archive preserves the literal prompt, input image, model revision and quantization settings. Judge editing quality separately from successful execution.")
                report = gr.JSON(label="Executed FLUX request and runtime", elem_id="flux-report")
    controls = [operation, prompt, width, height, seed, original, fit_reference]

    def request(mode, text, w, h, number, fit=False):
        return FluxRequest(mode, text, integer(w, "Width"), integer(h, "Height"), integer(number, "Seed"), fit)

    def association(mode, text, w, h, number, image, fit, previous):
        if previous is None:
            return "No completed FLUX run yet."
        try:
            current = request_digest(request(mode, text, w, h, number, fit), image)
        except ValueError:
            current = None
        prefix = "Completed" if current == previous["digest"] else "Settings changed — showing previous result"
        return f"{prefix} · run `{previous['run_id']}`"

    operation.input(lambda mode: gr.update(visible=mode == "Edit image"), operation, reference_group,
                    queue=False, api_name=False)
    example.click(lambda: (coffee.copy(), "Change the ceramic cup to blue. Keep the coffee and wooden table.", True),
                  outputs=[original, prompt, fit_reference], queue=False, api_name=False)

    def preview(image, enabled):
        if image is None:
            return None, "Upload a reference image."
        source = normalize_image(image)
        fit = ImageFit.plan(source.size, enabled=enabled, alignment=16)
        return fit.image(source), fit.summary()

    original.change(preview, [original, fit_reference], [reference_preview, size_info], queue=False, api_name=False)
    fit_reference.change(preview, [original, fit_reference], [reference_preview, size_info], queue=False, api_name=False)
    for control in controls:
        control.change(association, controls + [saved], badge, queue=False, api_name=False)

    outputs = [result_image, status, report, png, report_file, archive, saved]

    def execute(mode, text, w, h, number, image, fit, progress=gr.Progress()):
        try:
            result = service.run(request(mode, text, w, h, number, fit), image,
                                 progress=lambda fraction, message: progress(fraction, desc=message))
            return (result.image, result.status, result.report, result.result_file, result.report_file,
                    result.archive_file, {"digest": result.digest, "run_id": result.run_id})
        except Exception as exc:
            logger.exception("FLUX reference request failed")
            retained = [gr.skip() for _ in outputs]
            retained[1] = f"Failed: {type(exc).__name__}: {exc}. Previous completed result, if any, was retained."
            return tuple(retained)

    run.click(execute, controls, outputs, api_name="flux_run", concurrency_limit=1,
              concurrency_id="inference").then(association, controls + [saved], badge,
                                                queue=False, api_name=False)
