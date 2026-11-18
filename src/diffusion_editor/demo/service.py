"""Lazy model reuse, validated demo requests and reproducible local run artifacts."""
from collections.abc import Callable
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4
import zipfile

from PIL import Image
import torch

from diffusion_editor.config import ProjectConfig
from diffusion_editor.demo.image_fit import PreparedUpload, image_digest, prepare_upload, resolve_fitted_mask
from diffusion_editor.demo.ui_state import CONTROL, DDIM, EDIT, MODES, VAE
from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.evaluation.metrics import background_metrics, reconstruction_metrics
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
from diffusion_editor.pipelines.generation import TextToImagePipeline
from diffusion_editor.utils.environment import inspect_environment
from diffusion_editor.utils.image_utils import image_to_tensor, tensor_to_images
from diffusion_editor.utils.visualization import save_image_grid


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 50
    guidance: float = 7.5
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Enter a generation prompt.")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Negative prompt must be text.")
        if any(type(size) is not int or not 64 <= size <= 1024 or size % 8
               for size in (self.width, self.height)):
            raise ValueError("Generation width and height must be multiples of 8 in [64,1024].")
        if type(self.steps) is not int or not 1 <= self.steps <= 200:
            raise ValueError("Steps must be an integer between 1 and 200.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32).")
        validate_guidance_scale(self.guidance)


@dataclass
class GenerationDemoResult:
    image: Image.Image
    status: str
    report: dict[str, Any]
    result_file: str
    archive_file: str

@dataclass(frozen=True)
class DemoRequest:
    mode: str = "Local edit"
    source_prompt: str = "a photo of a white ceramic cup filled with coffee on a wooden table"
    target_prompt: str = "a black and white pencil drawing of a cup of coffee, detailed sketch, pencil shading on white paper"
    steps: int = 50
    strength: float = 1.0
    guidance: float = 7.5
    seed: int = 42
    mask_source: str = "Upload"
    soft_mask: bool = False
    feather_radius: float = 0
    latent_blending: bool = True
    preserve_background: bool = True
    inversion_guidance: float = 1.0
    negative_prompt: str = ""
    include_reconstruction: bool | None = None
    include_vae_reconstruction: bool = False
    include_edit: bool | None = None
    include_source_control: bool = False
    refine_reconstruction: bool = True
    show_intermediates: bool = False
    auto_fit: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("Choose Local edit, Global edit or Reconstruct source.")
        if self.include_edit is None:
            object.__setattr__(self, "include_edit", self.mode != "Reconstruct source")
        if self.include_reconstruction is None:
            object.__setattr__(self, "include_reconstruction", self.mode == "Reconstruct source")
        for field in ("soft_mask", "latent_blending", "preserve_background", "include_reconstruction",
                      "include_vae_reconstruction", "include_edit", "include_source_control",
                      "refine_reconstruction", "show_intermediates", "auto_fit"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean.")
        if not self.requested_outputs:
            raise ValueError("Select at least one output to calculate.")
        if self.mode == "Reconstruct source" and (self.include_edit or self.include_source_control):
            raise ValueError("Reconstruct source supports VAE and DDIM reconstruction outputs only.")
        diffusion = self.include_edit or self.include_source_control or self.include_reconstruction
        if not isinstance(self.source_prompt, str) or (diffusion and not self.source_prompt.strip()):
            raise ValueError("Enter a source prompt describing the original image.")
        if self.include_edit and (not isinstance(self.target_prompt, str) or not self.target_prompt.strip()):
            raise ValueError("Enter a target prompt describing the desired change.")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Negative prompt must be text.")
        if type(self.steps) is not int or not 1 <= self.steps <= 200:
            raise ValueError("Steps must be an integer between 1 and 200.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32).")
        if isinstance(self.strength, bool) or not isinstance(self.strength, Real) or not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Strength must be in [0,1].")
        validate_guidance_scale(self.guidance)
        validate_guidance_scale(self.inversion_guidance)
        if isinstance(self.feather_radius, bool) or not isinstance(self.feather_radius, Real) or not math.isfinite(self.feather_radius) or self.feather_radius < 0:
            raise ValueError("Feather radius must be finite and nonnegative.")
        if self.mask_source not in ("Upload", "Draw"):
            raise ValueError("Mask source must be Upload or Draw.")

    @property
    def requested_outputs(self) -> list[str]:
        return [label for label, enabled in ((EDIT, self.include_edit), (VAE, self.include_vae_reconstruction),
                (DDIM, self.include_reconstruction), (CONTROL, self.include_source_control)) if enabled]


@dataclass
class DemoResult:
    original: Image.Image
    vae: Image.Image | None
    reconstruction: Image.Image | None
    control: Image.Image | None
    edited: Image.Image | None
    mask: Image.Image | None
    previews: list[tuple[Image.Image, str]]
    status: str
    report: dict[str, Any]
    result_file: str
    archive_file: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


class DemoService:
    """One cached backend, serial inference and isolated output directories per run."""

    def __init__(self, config: ProjectConfig, output_dir: str | Path = "outputs/demo",
                 *, backend_loader: Callable[[ProjectConfig], SDBackend] | None = None) -> None:
        if not config.local_files_only:
            raise ValueError("The demo requires local_files_only=true. Cache model weights before launching.")
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self._loader = backend_loader or SDBackend.load
        self._backend = None
        self._editor = None
        self._reconstructor = None
        self._lock = threading.Lock()

    def _models(self) -> None:
        if self._backend is None:
            backend = self._loader(self.config)
            editor, reconstructor = EditingPipeline(backend), ReconstructionPipeline(backend)
            self._backend, self._editor, self._reconstructor = backend, editor, reconstructor

    def generate(self, request: GenerationRequest,
                 progress: Callable[[float, str], None] | None = None) -> GenerationDemoResult:
        """Independent txt2img entrypoint sharing the editing model cache and lock."""
        notify = progress or (lambda fraction, message: None)
        with self._lock:
            notify(0.05, "Loading cached model components")
            started = time.perf_counter()
            first_load = self._backend is None
            self._models()
            load_seconds = time.perf_counter() - started
            pipeline = TextToImagePipeline(self._backend)
            if self._backend.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            notify(0.15, "Generating with custom DDIM sampling and CFG")
            started = time.perf_counter()
            result = pipeline.generate(request.prompt, negative_prompt=request.negative_prompt,
                width=request.width, height=request.height, num_steps=request.steps,
                guidance_scale=request.guidance, seed=request.seed)
            if self._backend.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            image = tensor_to_images(result.image)[0]
            directory = self.output_dir / uuid4().hex
            directory.mkdir(parents=True)
            notify(0.9, "Saving generated image and report")
            image.save(directory / "generated.png")
            report = _json_safe({"mode": "txt2img", "method": "custom_ddim_eta0_cfg",
                "config": self.config.to_dict(), "request": asdict(request),
                "model_first_load": first_load, "model_load_seconds": load_seconds,
                "compute_seconds": elapsed, "sampling_timesteps": result.sampling.timesteps,
                "initial_noise": {"generator_device": "cpu", "dtype": "torch.float32",
                    "shape": list(result.initial_latent.shape),
                    "init_noise_sigma": float(self._backend.scheduler.init_noise_sigma),
                    "sha256": hashlib.sha256(result.initial_latent.numpy().tobytes()).hexdigest(),
                    "convention": "Gaussian prior at highest selected actual scheduler level; no source inversion"},
                "runtime": {"device": str(self._backend.device),
                    "unet_dtype": str(next(self._backend.unet.parameters()).dtype),
                    "vae_dtype": str(next(self._backend.vae.parameters()).dtype),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self._backend.device == "cuda" else None},
                "output_png_sha256": hashlib.sha256((directory / "generated.png").read_bytes()).hexdigest(),
                "metrics": None, "metric_note": "No source reference: PSNR, SSIM and background MAE are not applicable.",
                "environment": inspect_environment()})
            (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            archive = directory / "run.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                package.write(directory / "generated.png", "generated.png")
                package.write(directory / "report.json", "report.json")
            status = f"Text to image complete. Compute: {elapsed:.2f}s; model loading: {load_seconds:.2f}s. Output: {request.width} x {request.height}."
            notify(1.0, "Complete")
            return GenerationDemoResult(image, status, report, str(directory / "generated.png"), str(archive))

    def run(self, image: Image.Image | None, uploaded_mask: Image.Image | None,
            drawing: dict[str, Any] | None, request: DemoRequest,
            progress: Callable[[float, str], None] | None = None,
            *, input_state: PreparedUpload | None = None) -> DemoResult:
        notify = progress or (lambda fraction, text: None)
        prepared = prepare_upload(image, input_state, enabled=request.auto_fit)
        original = prepared.image
        if min(original.size) < 8 or max(original.size) > 1024:
            raise ValueError("Demo images must be between 8 and 1024 pixels on each side; around 512 is recommended.")
        run_edit = request.include_edit or request.include_source_control
        mask = raw_mask = mask_fit = None
        if request.mode == "Local edit" and run_edit:
            mask, raw_mask, mask_fit = resolve_fitted_mask(prepared, request.mask_source, uploaded_mask, drawing,
                                                        binary=not request.soft_mask)
        if request.feather_radius > max(original.size):
            raise ValueError("Feather radius is larger than the image.")
        with self._lock:
            notify(0.05, "Loading cached model components")
            started = time.perf_counter()
            first_load = self._backend is None
            self._models()
            model_load_seconds = time.perf_counter() - started
            factor = self._editor.vae.spatial_factor
            if original.width % factor or original.height % factor:
                raise ValueError(f"Unsupported image size {original.size}. Width and height must be multiples of {factor}; resize image and mask together.")
            torch.manual_seed(request.seed)
            if self._backend.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            compute_start = time.perf_counter()
            edit = reconstruction = None
            if run_edit:
                notify(0.15, "Running custom source inversion and editing")
                options = dict(strength=request.strength, guidance_scale=request.guidance, num_steps=request.steps,
                               inversion_guidance_scale=request.inversion_guidance, negative_prompt=request.negative_prompt,
                               return_trajectory=request.show_intermediates,
                               include_vae_reconstruction=request.include_vae_reconstruction,
                               include_source_control=request.include_source_control, include_edit=request.include_edit)
                if mask is None:
                    edit = self._editor.edit_global(original, request.source_prompt, request.target_prompt, **options)
                else:
                    edit = self._editor.edit_local(original, request.source_prompt, request.target_prompt, mask,
                                                   binary_mask=not request.soft_mask, feather_radius=request.feather_radius,
                                                   latent_blending=request.latent_blending,
                                                   preserve_background=request.preserve_background, **options)
            if request.include_reconstruction:
                notify(0.55, "Calculating independent DDIM source reconstruction")
                reconstruction = self._reconstructor.reconstruct(
                    original, request.source_prompt, request.steps, 1,
                    inversion_guidance_scale=1, negative_prompt=request.negative_prompt,
                    inversion_refinement_steps=12 if request.refine_reconstruction else 0,
                    return_trajectory=request.show_intermediates,
                    include_vae_reconstruction=request.include_vae_reconstruction and edit is None)
            tensors = {"original": image_to_tensor(original)}
            base = edit or reconstruction
            if base is not None:
                tensors["original"] = base.original
            if request.include_vae_reconstruction:
                if base is not None:
                    tensors["vae"] = base.vae_reconstruction
                else:
                    notify(0.2, "Calculating VAE reconstruction")
                    tensors["vae"] = self._editor.vae.decode_latents(self._editor.vae.encode_image(original))
            if self._backend.device == "cuda":
                torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_start
            if edit:
                if request.include_source_control:
                    tensors["control"] = edit.source_reconstruction
                if request.include_edit:
                    tensors["edited"] = edit.edited_image
            if reconstruction:
                tensors["reconstruction"] = reconstruction.ddim_reconstruction
            if mask is not None:
                tensors["mask"] = edit.pixel_mask.expand(-1, 3, -1, -1)
            images = {name: tensor_to_images(tensor)[0] for name, tensor in tensors.items()}
            metrics = {}
            for name in ("vae", "reconstruction", "control", "edited"):
                if name in tensors:
                    metrics[name] = reconstruction_metrics(tensors["original"], tensors[name])
            background = None
            if mask is not None:
                requested_mask = prepare_mask(mask, image_size=(original.height, original.width),
                                              latent_size=edit.latent_mask.shape[-2:], binary=not request.soft_mask)
                raw = edit.uncomposited_image if request.include_edit else edit.uncomposited_source
                final = edit.edited_image if request.include_edit else edit.source_reconstruction
                background = {"before_pixel_composite": background_metrics(edit.original, raw, edit.pixel_mask),
                              "final": background_metrics(edit.original, final, edit.pixel_mask),
                              "requested_mask_final": background_metrics(edit.original, final, requested_mask.pixel)}
            previews = []
            sampling = (edit.target_sampling or edit.source_sampling) if edit else reconstruction.sampling if reconstruction else None
            if request.show_intermediates and sampling is not None and sampling.trajectory:
                levels = sampling.timesteps
                for index in dict.fromkeys((0, len(levels) // 2, len(levels) - 1)):
                    t = levels[index]
                    preview = self._editor.vae.decode_latents(sampling.trajectory[t])
                    previews.append((tensor_to_images(preview)[0], f"Decoded latent t={t} (not a clean prediction)"))
            notify(0.9, "Saving images and reproducibility report")
            directory = self.output_dir / uuid4().hex
            directory.mkdir(parents=True)
            for name, value in images.items():
                value.save(directory / f"{name}.png")
            if prepared.fit.changed:
                prepared.original.save(directory / "uploaded_original.png")
            if mask is not None:
                mask.save(directory / "input_mask.png")
                if image_digest(raw_mask.convert("L")) != image_digest(mask):
                    raw_mask.save(directory / "mask_before_fit.png")
                if request.include_edit:
                    tensor_to_images(edit.uncomposited_image)[0].save(directory / "edited_uncomposited.png")
                if request.include_source_control:
                    tensor_to_images(edit.uncomposited_source)[0].save(directory / "control_uncomposited.png")
                torch.save({"pixel": edit.pixel_mask.cpu(), "latent": edit.latent_mask.cpu()}, directory / "mask_tensors.pt")
            for index, (value, label) in enumerate(previews):
                value.save(directory / f"preview_{index}.png")
            save_image_grid(list(tensors.values()), [name.replace("_", " ").title() for name in tensors],
                            directory / "comparison.png", columns=3)
            fidelity = reconstruction.fidelity_check if reconstruction else None
            with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("stage", "metric", "value"))
                for stage, values in metrics.items():
                    for metric, value in values.items():
                        writer.writerow((stage, metric, value))
                for stage, values in (background or {}).items():
                    for metric, value in values.items():
                        writer.writerow((f"background_{stage}", metric, value))
            report = _json_safe({"config": self.config.to_dict(), "request": asdict(request),
                      "computed_outputs": request.requested_outputs,
                      "preprocessing": prepared.fit.to_dict() | {"original_pixels_sha256": prepared.original_digest,
                                          "canvas_pixels_sha256": prepared.canvas_digest, "mask_transform": mask_fit},
                      "metric_domain": "fitted_model_canvas_including_padding",
                      "model_first_load": first_load, "model_load_seconds": model_load_seconds,
                      "compute_seconds": compute_seconds,
                      "runtime": {"device": str(self._backend.device), "unet_dtype": str(next(self._backend.unet.parameters()).dtype),
                                  "vae_dtype": str(next(self._backend.vae.parameters()).dtype),
                                  "input_width": original.width, "input_height": original.height,
                                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self._backend.device == "cuda" else None},
                      "source_png_sha256": hashlib.sha256((directory / "original.png").read_bytes()).hexdigest(),
                      "mask_png_sha256": hashlib.sha256((directory / "input_mask.png").read_bytes()).hexdigest() if mask else None,
                      "selection": asdict(edit.selection) if edit else None,
                      "source_timesteps": edit.source_sampling.timesteps if edit and edit.source_sampling else None,
                      "target_timesteps": edit.target_sampling.timesteps if edit and edit.target_sampling else None,
                      "reconstruction_timesteps": reconstruction.sampling.timesteps if reconstruction else None,
                      "reconstruction_method": fidelity.selected_method if fidelity else "plain_ddim" if reconstruction else None,
                      "fidelity_check": {"refined_latent_mse": fidelity.refined_latent_mse,
                                         "plain_latent_mse": fidelity.plain_latent_mse} if fidelity else None,
                      "reconstruction_refinement": {str(t): asdict(d) for t, d in fidelity.candidate_refinement.items()} if fidelity else None,
                      "metrics": metrics, "background_metrics": background,
                      "background_output": ("edited" if request.include_edit else "control") if mask is not None else None,
                      "metric_settings": {"data_range": 1, "color": "RGB", "ssim_window": 7,
                                          "ssim_gaussian_weights": False, "ssim_sample_covariance": True},
                      "metric_note": "PSNR/SSIM measure preservation, not prompt alignment. Null PSNR means exact match or undefined empty region; inspect region weight.",
                      "environment": inspect_environment()})
            (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            archive = directory / "run.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                for file in directory.iterdir():
                    if file != archive and file.is_file():
                        package.write(file, file.name)
            primary = directory / next(f"{name}.png" for name in ("edited", "reconstruction", "vae", "control") if name in images)
            status = f"{request.mode} complete. Outputs: {', '.join(request.requested_outputs)}. Compute: {compute_seconds:.2f}s; model loading: {model_load_seconds:.2f}s."
            if edit:
                status += f" Start timestep {edit.selection.start_timestep}; reverse steps {edit.selection.reverse_steps}."
            status += f" Canvas/output: {original.width} x {original.height}."
            notify(1.0, "Complete")
            return DemoResult(images["original"], images.get("vae"), images.get("reconstruction"), images.get("control"),
                              images.get("edited"), images.get("mask"), previews, status, report, str(primary), str(archive))
