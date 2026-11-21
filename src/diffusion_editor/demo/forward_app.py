"""Tab-local forward-noising controls backed by the shared custom-model service."""
import logging
import math
from typing import Any

from PIL import Image

from diffusion_editor.demo.inputs import editor_canvas
from diffusion_editor.demo.forward_service import ForwardRequest


def build_forward_tab(gr: Any, service: Any, mode: str, coffee: Image.Image, coffee_mask: Image.Image) -> None:
    """Called inside a persistent Gradio tab; image fitting is explicit, not implicit."""
    gr.Markdown("Forward noising → custom DDIM/CFG. No source inversion or source prompt. "
                "Use dimensions divisible by 8; final canvas must be ≤1024 pixels per side. "
                "Strength uses floor(strength × steps); zero performs no reverse updates. "
                "VAE decoding and the selected pixel compositing still apply.")
    source = gr.Image(value=coffee, type="pil", image_mode="RGB", format="png", sources=["upload"],
                      interactive=True, label=f"{mode}: source image", height=260)
    prompt = gr.Textbox(value="a black and white pencil drawing of a cup of coffee" if mode != "outpaint"
                        else "a photo of a cup of coffee on a wooden table, wider view, continuous wooden tabletop",
                        label=f"{mode}: target prompt", lines=2)
    negative = gr.Textbox(value="", label="Negative prompt")
    if mode == "inpaint":
        mask_source = gr.Radio(["Upload", "Draw"], value="Upload", label="Mask source")
        mask = gr.Image(value=coffee_mask, type="pil", image_mode="L", format="png", sources=["upload"],
                        label="Inpainting mask: white edits, black preserves", height=200, interactive=True)
        drawing = gr.ImageEditor(value=editor_canvas(coffee), type="pil", image_mode="RGBA", format="png",
            sources=(), transforms=(), layers=False, interactive=True, height=320,
            brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed", default_size=30),
            eraser=gr.Eraser(default_size=30), label="Paint the edit region (used when Draw is selected)")
        clear = gr.Button("Reset inpainting drawing")
        soft = gr.Checkbox(False, label="Soft uploaded/painted mask weights")
    else:
        mask_source, mask, drawing, soft = gr.State("Upload"), gr.State(None), gr.State(None), gr.State(False)
    if mode == "outpaint":
        gr.Markdown("Margins add canvas pixels without resizing the source. Default overlap 0 preserves the "
                    "entire source rectangle with compositing enabled. Overlap or feathering permits source-border changes.")
        with gr.Row():
            left = gr.Number(0, precision=0, label="Expand left (px)")
            right = gr.Number(128, precision=0, label="Expand right (px)")
            top = gr.Number(0, precision=0, label="Expand top (px)")
            bottom = gr.Number(0, precision=0, label="Expand bottom (px)")
        overlap = gr.Number(0, precision=0, label="Editable overlap inside expanded source edges (px)")
        fill = gr.Radio(["edge", "neutral"], value="edge", label="Canvas initialization")
    else:
        left, right, top, bottom, overlap, fill = (gr.State(v) for v in (0, 0, 0, 0, 0, "edge"))
    with gr.Row():
        steps = gr.Slider(1, 200, value=50, step=1, label="Forward-route steps")
        strength = gr.Slider(0, 1, value=1 if mode == "outpaint" else 0.85, step=0.05, label="Forward-noising strength")
        guidance = gr.Slider(0, 20, value=7.5, step=0.5, label="Target guidance")
        seed = gr.Number(42, precision=0, label="Noise seed", info="Changes the sampled forward noise.")
    if mode != "img2img":
        feather = gr.Slider(0, 32, value=0, step=1, label="Mask feather sigma (source pixels)")
        latent = gr.Checkbox(True, label="Blend known-region noised latents at each updated level")
        preserve = gr.Checkbox(True, label="Composite original pixels wherever effective mask is zero")
    else:
        feather, latent, preserve = gr.State(0), gr.State(True), gr.State(True)
    run = gr.Button("Run " + mode, variant="primary")
    status = gr.Textbox(label="Forward-route status", interactive=False)
    with gr.Row():
        canvas_out = gr.Image(type="pil", format="png", label="Input / initialized canvas", interactive=False)
        edited_out = gr.Image(type="pil", format="png", label="Forward-noising result", interactive=False)
    with gr.Accordion("Raw decoded edit and effective mask", open=False):
        raw_out = gr.Image(type="pil", format="png", label="Before pixel compositing", interactive=False)
        mask_out = gr.Image(type="pil", format="png", label="Effective edit mask", interactive=False)
    with gr.Row():
        png = gr.File(label="Download forward result PNG", interactive=False)
        archive = gr.File(label="Download forward run archive", interactive=False)
    report = gr.JSON(label="Forward noising settings, metrics and runtime")

    def integer(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise ValueError(f"{label} must be an integer")
        return int(value)

    def execute(image, mask_image, paint, target_prompt, negative_prompt, num_steps, edit_strength,
                edit_guidance, random_seed, mask_choice, soft_weights, feather_pixels, blend_latents,
                preserve_pixels, margin_left, margin_right, margin_top, margin_bottom, overlap_pixels,
                canvas_fill, progress=gr.Progress()):
        try:
            request = ForwardRequest(mode, target_prompt, negative_prompt=negative_prompt,
                steps=integer(num_steps, "Steps"), strength=edit_strength, guidance=edit_guidance,
                seed=integer(random_seed, "Seed"), mask_source=mask_choice, soft_mask=soft_weights,
                feather_radius=feather_pixels, latent_blending=blend_latents, preserve_background=preserve_pixels,
                left=integer(margin_left, "Left"), right=integer(margin_right, "Right"),
                top=integer(margin_top, "Top"), bottom=integer(margin_bottom, "Bottom"),
                overlap=integer(overlap_pixels, "Overlap"), fill=canvas_fill)
            result = service.forward_edit(image, mask_image, paint, request,
                progress=lambda fraction, message: progress(fraction, desc=message))
            return result.canvas, result.edited, result.raw, result.mask, result.status, result.result_file, result.archive_file, result.report
        except Exception as exc:
            logging.getLogger(__name__).exception("Forward-noising operation failed")
            raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    inputs = [source, mask, drawing, prompt, negative, steps, strength, guidance, seed, mask_source,
              soft, feather, latent, preserve, left, right, top, bottom, overlap, fill]
    outputs = [canvas_out, edited_out, raw_out, mask_out, status, png, archive, report]
    run.click(execute, inputs, outputs, api_name="forward_" + mode, concurrency_limit=1, concurrency_id="inference")

    def reset(image):
        return (editor_canvas(image), None, None, None, None, None, "Input changed; draw/upload an aligned mask and rerun.", None, None, None)
    source.upload(reset, source, [drawing, mask] + outputs, queue=False, api_name=False)
    source.clear(reset, source, [drawing, mask] + outputs, queue=False, api_name=False)
    if mode == "inpaint":
        clear.click(lambda image: (editor_canvas(image), "Draw"), source, [drawing, mask_source], queue=False, api_name=False)
    gr.Markdown("This is base-model masked sampling. Seams, duplicated objects and weak prompt adherence can remain. "
                "PSNR/SSIM do not establish semantic success; outpainted borders have no ground-truth reference.")
