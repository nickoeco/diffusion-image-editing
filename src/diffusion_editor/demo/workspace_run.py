"""Task routing adapters. All inference stays in the existing shared service."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import zipfile

from PIL import Image

from diffusion_editor.demo.image_fit import prepare_upload, resolve_fitted_mask
from diffusion_editor.demo.service import DemoRequest, GenerationRequest
from diffusion_editor.demo.forward_service import ForwardRequest
from diffusion_editor.demo.ui_state import CONTROL, DDIM, EDIT, VAE, validate_outputs
from diffusion_editor.demo.workspace_state import integer, settings_digest


@dataclass
class WorkspaceResult:
    before: Image.Image | None
    image: Image.Image
    requested_mask: Image.Image | None
    effective_mask: Image.Image | None
    diagnostics: list
    status: str
    report: dict
    result_file: str
    archive_file: str
    source_prompt: str
    target_prompt: str

    @property
    def run_id(self) -> str:
        return Path(self.result_file).parent.name


def execute_task(service: Any, task: str, values: dict, notify: Callable) -> WorkspaceResult:
    """Keep UI provenance alongside, without rewriting the backend report/evidence."""
    result = _execute_task(service, task, values, notify)
    directory = Path(result.result_file).parent
    manifest = {"task": task, "run_id": result.run_id, "settings_digest": settings_digest(values),
                "source_prompt": result.source_prompt, "target_prompt": result.target_prompt,
                "output_png_sha256": hashlib.sha256(Path(result.result_file).read_bytes()).hexdigest(),
                "generation_request": result.report["request"],
                "ui_preprocessing": result.report.get("workspace"),
                "note": "UI provenance; report.json retains the backend's numerical evidence."}
    files = []
    if task != "generate":
        prepared = prepare_upload(values["image"], values.get("prepared"), enabled=values["fit"])
        manifest["input_sha256"] = {"original_pixels": prepared.original_digest, "canvas_pixels": prepared.canvas_digest}
        manifest["preprocessing"] = prepared.fit.to_dict()
        if prepared.fit.changed:
            original = directory / "workspace_original_upload.png"
            prepared.original.save(original)
            files.append(original)
    sidecar = directory / "workspace.json"
    sidecar.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    files.append(sidecar)
    with zipfile.ZipFile(result.archive_file, "a", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.name)
    result.report = dict(result.report) | {"workspace_association": manifest}
    return result


def _execute_task(service: Any, task: str, values: dict, notify: Callable) -> WorkspaceResult:
    """Compute only the task/outputs requested, with explicit image preparation."""
    if task not in ("generate", "edit", "extend", "diagnostics"):
        raise ValueError("Unknown workspace task.")
    if task == "edit" and (values["scope"] not in ("Whole image", "Selected region") or
                           values["method"] not in ("Forward noising", "DDIM inversion")):
        raise ValueError("Choose a valid edit area and editing method.")
    common = dict(steps=integer(values["steps"], "Steps"), guidance=values["guidance"],
                  seed=integer(values["seed"], "Seed"), negative_prompt=values["negative"])
    if task == "generate":
        result = service.generate(GenerationRequest(prompt=values["prompt"],
            width=integer(values["width"], "Width"), height=integer(values["height"], "Height"), **common), progress=notify)
        return WorkspaceResult(None, result.image, None, None, [], result.status, result.report,
                               result.result_file, result.archive_file, "", values["prompt"])
    prepared = prepare_upload(values["image"], values.get("prepared"), enabled=values["fit"])
    local = task == "edit" and values["scope"] == "Selected region"
    masking = dict(mask_source=values["mask_source"], soft_mask=values["soft"],
                   feather_radius=values["feather"], latent_blending=values["latent"],
                   preserve_background=values["preserve"])
    if task == "extend" or task == "edit" and values["method"] == "Forward noising":
        mask = None
        if local:
            mask, _, _ = resolve_fitted_mask(prepared, values["mask_source"], values["uploaded_mask"],
                                             values["paint"], binary=not values["soft"])
        request = ForwardRequest(mode="outpaint" if task == "extend" else "inpaint" if local else "img2img",
            prompt=values["prompt"], strength=values["strength"], **common, **masking,
            **{key: integer(values[key], key.title()) for key in ("left", "right", "top", "bottom", "overlap")},
            fill=values["fill"])
        # Resolve Draw against the prepared surface once, then pass explicit aligned weights.
        result = service.forward_edit(prepared.image, mask, None, replace(request, mask_source="Upload"), progress=notify)
        requested = mask
        if task == "extend":
            from diffusion_editor.demo.workspace_state import outpaint_preview
            _, requested, _ = outpaint_preview(prepared.image, request.left, request.right, request.top,
                                               request.bottom, request.overlap, request.fill)
        report = dict(result.report)
        report["workspace"] = {"preprocessing": prepared.fit.to_dict(), "original_pixels_sha256": prepared.original_digest,
                               "canvas_pixels_sha256": prepared.canvas_digest,
                               "mask_input": values["mask_source"] if local else None}
        return WorkspaceResult(result.canvas, result.edited, requested, result.mask,
            [(result.raw, "Before pixel compositing")], result.status, report, result.result_file,
            result.archive_file, "", values["prompt"])
    if task not in ("edit", "diagnostics"):
        raise ValueError("Unknown workspace task.")
    operation = "Local edit" if local else "Global edit" if task == "edit" or CONTROL in values["selected_outputs"] else "Reconstruct source"
    chosen = [EDIT] if task == "edit" else values["selected_outputs"]
    validate_outputs(operation, chosen)
    request = DemoRequest(mode=operation, source_prompt=values["source_prompt"], target_prompt=values["prompt"],
        strength=values["strength"], **common, **masking, auto_fit=values["fit"],
        inversion_guidance=values["inversion_guidance"], include_edit=EDIT in chosen,
        include_reconstruction=DDIM in chosen, include_vae_reconstruction=VAE in chosen,
        include_source_control=CONTROL in chosen, refine_reconstruction=values["refine"],
        show_intermediates=values["intermediates"])
    result = service.run(prepared.image, values["uploaded_mask"], values["paint"], request,
                         progress=notify, input_state=prepared)
    image = result.edited if EDIT in chosen else result.reconstruction if DDIM in chosen else result.vae if VAE in chosen else result.control
    requested = None
    if local:
        requested, _, _ = resolve_fitted_mask(prepared, values["mask_source"], values["uploaded_mask"],
                                             values["paint"], binary=not values["soft"])
    extra = [(value, label) for value, label in ((result.vae, VAE), (result.reconstruction, DDIM),
              (result.control, CONTROL)) if value is not None]
    return WorkspaceResult(result.original, image, requested, result.mask, extra + result.previews,
        result.status, result.report, result.result_file, result.archive_file,
        values["source_prompt"], values["prompt"] if task == "edit" else "")
