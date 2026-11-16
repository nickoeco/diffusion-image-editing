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
from diffusion_editor.demo.inputs import normalize_image, resolve_mask
from diffusion_editor.diffusion.guidance import validate_guidance_scale
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.evaluation.metrics import background_metrics, reconstruction_metrics
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
from diffusion_editor.utils.environment import inspect_environment
from diffusion_editor.utils.image_utils import tensor_to_images
from diffusion_editor.utils.visualization import save_image_grid

MODES = ("Local edit", "Global edit", "Reconstruct source")


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
    include_reconstruction: bool = True
    refine_reconstruction: bool = True
    show_intermediates: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("Choose Local edit, Global edit or Reconstruct source.")
        if not isinstance(self.source_prompt, str) or not self.source_prompt.strip():
            raise ValueError("Enter a source prompt describing the original image.")
        if self.mode != "Reconstruct source" and (not isinstance(self.target_prompt, str) or not self.target_prompt.strip()):
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
        for field in ("soft_mask", "latent_blending", "preserve_background", "include_reconstruction",
                      "refine_reconstruction", "show_intermediates"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean.")


@dataclass
class DemoResult:
    original: Image.Image
    vae: Image.Image
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

    def run(self, image: Image.Image | None, uploaded_mask: Image.Image | None,
            drawing: dict[str, Any] | None, request: DemoRequest,
            progress: Callable[[float, str], None] | None = None) -> DemoResult:
        notify = progress or (lambda fraction, text: None)
        original = normalize_image(image)
        if min(original.size) < 8 or max(original.size) > 1024:
            raise ValueError("Demo images must be between 8 and 1024 pixels on each side; around 512 is recommended.")
        mask = resolve_mask(original, request.mask_source, uploaded_mask, drawing) if request.mode == "Local edit" else None
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
            if request.mode != "Reconstruct source":
                notify(0.15, "Running custom source inversion and editing")
                options = dict(strength=request.strength, guidance_scale=request.guidance, num_steps=request.steps,
                               inversion_guidance_scale=request.inversion_guidance, negative_prompt=request.negative_prompt,
                               return_trajectory=request.show_intermediates)
                if mask is None:
                    edit = self._editor.edit_global(original, request.source_prompt, request.target_prompt, **options)
                else:
                    edit = self._editor.edit_local(original, request.source_prompt, request.target_prompt, mask,
                                                   binary_mask=not request.soft_mask, feather_radius=request.feather_radius,
                                                   latent_blending=request.latent_blending,
                                                   preserve_background=request.preserve_background, **options)
            if request.mode == "Reconstruct source" or request.include_reconstruction:
                notify(0.55, "Calculating independent DDIM source reconstruction")
                reconstruction = self._reconstructor.reconstruct(
                    original, request.source_prompt, request.steps, 1,
                    inversion_guidance_scale=1, negative_prompt=request.negative_prompt,
                    inversion_refinement_steps=12 if request.refine_reconstruction else 0,
                    return_trajectory=request.show_intermediates)
            if self._backend.device == "cuda":
                torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_start
            base = edit or reconstruction
            tensors = {"original": base.original, "vae": base.vae_reconstruction}
            if edit:
                tensors.update(control=edit.source_reconstruction, edited=edit.edited_image)
            if reconstruction:
                tensors["reconstruction"] = reconstruction.ddim_reconstruction
            if mask is not None:
                tensors["mask"] = edit.pixel_mask.expand(-1, 3, -1, -1)
            images = {name: tensor_to_images(tensor)[0] for name, tensor in tensors.items()}
            metrics = {}
            for name in ("vae", "reconstruction", "control", "edited"):
                if name in tensors:
                    metrics[name] = reconstruction_metrics(base.original, tensors[name])
            background = None
            if mask is not None:
                requested_mask = prepare_mask(mask, image_size=(original.height, original.width),
                                              latent_size=edit.latent_mask.shape[-2:], binary=not request.soft_mask)
                background = {"before_pixel_composite": background_metrics(edit.original, edit.uncomposited_image, edit.pixel_mask),
                              "final": background_metrics(edit.original, edit.edited_image, edit.pixel_mask),
                              "requested_mask_final": background_metrics(edit.original, edit.edited_image, requested_mask.pixel)}
            previews = []
            sampling = edit.target_sampling if edit else reconstruction.sampling
            if request.show_intermediates and sampling.trajectory:
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
            if mask is not None:
                mask.save(directory / "input_mask.png")
                tensor_to_images(edit.uncomposited_image)[0].save(directory / "edited_uncomposited.png")
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
                      "model_first_load": first_load, "model_load_seconds": model_load_seconds,
                      "compute_seconds": compute_seconds,
                      "runtime": {"device": str(self._backend.device), "unet_dtype": str(next(self._backend.unet.parameters()).dtype),
                                  "vae_dtype": str(next(self._backend.vae.parameters()).dtype),
                                  "input_width": original.width, "input_height": original.height,
                                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self._backend.device == "cuda" else None},
                      "source_png_sha256": hashlib.sha256((directory / "original.png").read_bytes()).hexdigest(),
                      "mask_png_sha256": hashlib.sha256((directory / "input_mask.png").read_bytes()).hexdigest() if mask else None,
                      "selection": asdict(edit.selection) if edit else None,
                      "source_timesteps": edit.source_sampling.timesteps if edit else None,
                      "target_timesteps": edit.target_sampling.timesteps if edit else None,
                      "reconstruction_timesteps": reconstruction.sampling.timesteps if reconstruction else None,
                      "reconstruction_method": fidelity.selected_method if fidelity else "plain_ddim" if reconstruction else None,
                      "fidelity_check": {"refined_latent_mse": fidelity.refined_latent_mse,
                                         "plain_latent_mse": fidelity.plain_latent_mse} if fidelity else None,
                      "reconstruction_refinement": {str(t): asdict(d) for t, d in fidelity.candidate_refinement.items()} if fidelity else None,
                      "metrics": metrics, "background_metrics": background,
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
            primary = directory / ("edited.png" if edit else "reconstruction.png")
            status = f"{request.mode} complete. Compute: {compute_seconds:.2f}s; model loading: {model_load_seconds:.2f}s."
            if edit:
                status += f" Start timestep {edit.selection.start_timestep}; reverse steps {edit.selection.reverse_steps}."
            notify(1.0, "Complete")
            return DemoResult(images["original"], images["vae"], images.get("reconstruction"), images.get("control"),
                              images.get("edited"), images.get("mask"), previews, status, report, str(primary), str(archive))
