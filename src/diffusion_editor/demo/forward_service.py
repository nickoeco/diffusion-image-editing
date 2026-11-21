"""Forward-noising request validation and shared-cache demo/CLI orchestration."""
from collections.abc import Callable
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any
from uuid import uuid4
import zipfile

from PIL import Image
import torch

from diffusion_editor.demo.inputs import normalize_image, resolve_mask
from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.evaluation.metrics import background_metrics, reconstruction_metrics
from diffusion_editor.evaluation.records import write_json
from diffusion_editor.pipelines.forward_editing import ForwardEditingPipeline
from diffusion_editor.utils.environment import inspect_environment
from diffusion_editor.utils.image_utils import tensor_to_images
from diffusion_editor.utils.visualization import save_image_grid

FORWARD_MODES = ("img2img", "inpaint", "outpaint")


@dataclass(frozen=True)
class ForwardRequest:
    mode: str
    prompt: str
    negative_prompt: str = ""
    steps: int = 50
    strength: float = 0.85
    guidance: float = 7.5
    seed: int = 42
    mask_source: str = "Upload"
    soft_mask: bool = False
    feather_radius: float = 0
    latent_blending: bool = True
    preserve_background: bool = True
    left: int = 0
    right: int = 128
    top: int = 0
    bottom: int = 0
    overlap: int = 0
    fill: str = "edge"

    def __post_init__(self) -> None:
        if self.mode not in FORWARD_MODES or not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Choose img2img/inpaint/outpaint and enter a target prompt")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Negative prompt must be text")
        if type(self.steps) is not int or not 1 <= self.steps <= 200:
            raise ValueError("Steps must be an integer in [1,200]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32)")
        if isinstance(self.strength, bool) or not isinstance(self.strength, Real) or not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Strength must be finite in [0,1]")
        validate_guidance_scale(self.guidance)
        if isinstance(self.feather_radius, bool) or not isinstance(self.feather_radius, Real) or not math.isfinite(self.feather_radius) or not 0 <= self.feather_radius <= 64:
            raise ValueError("Feather sigma must be finite in [0,64]")
        if any(type(getattr(self, flag)) is not bool for flag in ("soft_mask", "latent_blending", "preserve_background")):
            raise ValueError("Mask and compositing switches must be boolean")
        if self.mask_source not in ("Upload", "Draw") or self.fill not in ("edge", "neutral"):
            raise ValueError("Choose a valid mask source and canvas fill")
        if any(type(value) is not int or value < 0 for value in (self.left, self.right, self.top, self.bottom, self.overlap)):
            raise ValueError("Margins and overlap must be nonnegative integers")
        if self.mode == "outpaint" and not any((self.left, self.right, self.top, self.bottom)):
            raise ValueError("Expand at least one side for outpainting")


@dataclass
class ForwardDemoResult:
    original: Image.Image
    canvas: Image.Image
    edited: Image.Image
    raw: Image.Image
    mask: Image.Image | None
    status: str
    report: dict[str, Any]
    result_file: str
    archive_file: str


def run_forward(owner: Any, image: Image.Image | None, mask: Image.Image | None, drawing: dict | None,
                request: ForwardRequest, progress: Callable[[float, str], None] | None = None) -> ForwardDemoResult:
    """Use DemoService's sole cache/lock. No inversion or calibration is called."""
    notify = progress or (lambda fraction, text: None)
    source = normalize_image(image)
    width = source.width + (request.left + request.right if request.mode == "outpaint" else 0)
    height = source.height + (request.top + request.bottom if request.mode == "outpaint" else 0)
    if min(source.size) < 8 or max(width, height) > 1024:
        raise ValueError("Source sides must be >=8 and final canvas sides <=1024 pixels; resize explicitly")
    if width % 8 or height % 8:
        raise ValueError("SD 1.5 image/final canvas dimensions must be multiples of 8; no automatic resize")
    if request.mode == "outpaint" and 2 * request.overlap >= min(source.size):
        raise ValueError("Overlap must be less than half the smaller source dimension")
    resolved = resolve_mask(source, request.mask_source, mask, drawing) if request.mode == "inpaint" else None
    with owner._lock:
        notify(0.05, "Loading cached model components")
        first_load = owner._backend is None
        started = time.perf_counter()
        owner._models()
        load_seconds = time.perf_counter() - started
        backend = owner._backend
        pipeline = ForwardEditingPipeline(backend)
        if backend.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        notify(0.15, "Forward noising and custom DDIM sampling")
        options = dict(strength=request.strength, guidance_scale=request.guidance, num_steps=request.steps,
                       seed=request.seed, negative_prompt=request.negative_prompt)
        started = time.perf_counter()
        if request.mode == "img2img":
            result = pipeline.img2img(source, request.prompt, **options)
        else:
            options.update(feather_radius=request.feather_radius, latent_blending=request.latent_blending,
                           preserve_background=request.preserve_background)
            if request.mode == "inpaint":
                result = pipeline.inpaint(source, resolved, request.prompt, binary_mask=not request.soft_mask, **options)
            else:
                result = pipeline.outpaint(source, request.prompt, left=request.left, right=request.right,
                    top=request.top, bottom=request.bottom, overlap=request.overlap, fill=request.fill, **options)
        if backend.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        notify(0.9, "Saving raw/final outputs and metrics")
        tensors = {"canvas": result.original, "vae": result.vae_reconstruction,
                   "raw": result.uncomposited_image, "edited": result.edited_image}
        if result.pixel_mask is not None:
            tensors["effective_mask"] = result.pixel_mask.expand(-1, 3, -1, -1)
        images = {key: tensor_to_images(value)[0] for key, value in tensors.items()}
        metrics = {key: reconstruction_metrics(result.original, tensors[key]) for key in ("vae", "raw", "edited")}
        background = ({"raw": background_metrics(result.original, result.uncomposited_image, result.pixel_mask),
                       "final": background_metrics(result.original, result.edited_image, result.pixel_mask)}
                      if result.pixel_mask is not None else None)
        requested_weights = None
        if result.pixel_mask is not None:
            requested = result.canvas.mask if result.canvas is not None else resolved
            requested_weights = prepare_mask(requested, tuple(result.original.shape[-2:]),
                tuple(result.latent_mask.shape[-2:]), binary=not request.soft_mask if request.mode == "inpaint" else False).pixel
            background.update(requested_raw=background_metrics(result.original, result.uncomposited_image, requested_weights),
                              requested_final=background_metrics(result.original, result.edited_image, requested_weights))
        retained_source = None
        if result.canvas is not None:
            left, top, right, bottom = result.canvas.source_box
            reference = result.original[:, :, top:bottom, left:right]
            retained_source = {"raw": reconstruction_metrics(reference, result.uncomposited_image[:, :, top:bottom, left:right]),
                               "final": reconstruction_metrics(reference, result.edited_image[:, :, top:bottom, left:right])}
        directory = owner.output_dir / uuid4().hex
        directory.mkdir(parents=True)
        source.save(directory / "original.png")
        for name, value in images.items():
            value.save(directory / f"{name}.png")
        if resolved is not None:
            resolved.save(directory / "requested_mask.png")
        if result.canvas is not None:
            tensor_to_images(result.canvas.mask.expand(-1, 3, -1, -1))[0].convert("L").save(directory / "requested_mask.png")
        if result.pixel_mask is not None:
            torch.save({"requested": requested_weights.cpu(), "pixel": result.pixel_mask.cpu(),
                        "latent": result.latent_mask.cpu()}, directory / "masks.pt")
        torch.save(result.noise, directory / "noise.pt")
        save_image_grid(list(tensors.values()), list(tensors), directory / "comparison.png")
        all_metrics = {**metrics, **{f"background_{k}": v for k, v in (background or {}).items()},
                       **{f"retained_source_{k}": v for k, v in (retained_source or {}).items()}}
        with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("stage", "metric", "value"))
            for stage, values in all_metrics.items():
                writer.writerows((stage, key, value) for key, value in values.items())
        report = {"method": "forward_noising_custom_ddim_eta0", "mode": request.mode,
            "config": owner.config.to_dict(), "request": asdict(request), "selection": asdict(result.selection),
            "strength_policy": "floor(strength*N); below 1/N selects clean with zero reverse updates",
            "sampling_timesteps": result.sampling.timesteps, "inversion_executed": False,
            "noise": {"seed": request.seed, "generator_device": "cpu", "dtype": "float32",
                      "shape": list(result.noise.shape), "sha256": hashlib.sha256(result.noise.numpy().tobytes()).hexdigest(),
                      "known_region_policy": "same epsilon, q(z_s|z0) at UPDATED timestep s; CLEAN is exact z0"},
            "canvas_layout": {"source_box_xyxy": result.canvas.source_box, "margins_lrtb": result.canvas.margins,
                              "overlap": result.canvas.overlap, "fill": result.canvas.fill} if result.canvas else None,
            "metrics": metrics, "background_metrics": background, "retained_source_metrics": retained_source,
            "metric_settings": {"data_range": 1, "color": "RGB", "ssim_window": 7,
                                "ssim_gaussian_weights": False, "ssim_sample_covariance": True,
                                "background_denominator": "3*sum(1-mask)", "null_psnr": "positive region weight or defined whole image: exact match; empty region: undefined"},
            "metric_note": "Scores measure preservation, not semantic success. For outpainting, full-canvas scores compare generated borders to a synthetic initialization, not ground truth. Judge new content and seams visually.",
            "model_first_load": first_load, "model_load_seconds": load_seconds, "compute_seconds": elapsed,
            "runtime": {"device": backend.device, "unet_dtype": str(next(backend.unet.parameters()).dtype),
                        "vae_dtype": str(next(backend.vae.parameters()).dtype),
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None},
            "file_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in directory.iterdir()},
            "environment": inspect_environment()}
        write_json(directory / "report.json", report)
        archive = directory / "run.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
            for file in directory.iterdir():
                if file != archive:
                    package.write(file, file.name)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        status = f"{request.mode} complete. {width} x {height}; start t={result.selection.start_timestep}, {result.selection.reverse_steps} reverse steps. Compute {elapsed:.2f}s; loading {load_seconds:.2f}s."
        notify(1, "Complete")
        return ForwardDemoResult(source, images["canvas"], images["edited"], images["raw"], images.get("effective_mask"),
                                 status, report, str(directory / "edited.png"), str(archive))
