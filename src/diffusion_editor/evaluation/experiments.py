"""Sequential custom-pipeline experiments with strict manifests and resumability."""
from dataclasses import asdict
from collections.abc import Callable
import json
import logging
from pathlib import Path
import shutil
import time

from PIL import Image, ImageOps
import torch

from diffusion_editor.config import ProjectConfig
from diffusion_editor.editing.mask import prepare_mask
from diffusion_editor.evaluation.artifacts import calculate_metrics, count_unet, save_case, sync
from diffusion_editor.evaluation.records import validate_completed, write_json
from diffusion_editor.evaluation.experiment_config import ExperimentSuite, file_hash, fingerprint, input_hashes
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
from diffusion_editor.utils.environment import inspect_environment

logger = logging.getLogger(__name__)


def code_snapshot(root: Path) -> dict[str, str]:
    files = sorted((root / "src/diffusion_editor").rglob("*.py"))
    entry = root / "scripts/run_experiments.py"
    if entry.exists():
        files.append(entry)
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in files}


class ExperimentRunner:
    """Load once; execute exactly one changed setting per study, without downloads."""

    def __init__(self, config: ProjectConfig, *, backend_loader: Callable | None = None,
                 environment_reader: Callable = inspect_environment) -> None:
        if not config.local_files_only:
            raise ValueError("Experiments require local_files_only=true; cache weights separately")
        self.config = config
        self._loader = backend_loader or SDBackend.load
        self._environment = environment_reader
        self.backend = None

    def run(self, suite: ExperimentSuite, output_dir: str | Path, *, root: str | Path = ".", resume: bool = False) -> list[dict]:
        root, output = Path(root).resolve(), Path(output_dir).resolve()
        hashes = input_hashes(suite, root)
        for sample in suite.samples:
            with Image.open(root / sample.image) as value:
                image_size = ImageOps.exif_transpose(value).size
            if min(image_size) < 8:
                raise ValueError("Experiment image sides must be at least 8 pixels")
            if sample.mask:
                with Image.open(root / sample.mask) as value:
                    mask = ImageOps.exif_transpose(value)
                    if mask.mode not in ("L", "1") or mask.size != image_size:
                        raise ValueError("Evaluation masks must be grayscale and exactly aligned with the image")
        environment = self._environment()
        identity = {"suite": suite.to_dict(), "model_config": self.config.to_dict(), "inputs": hashes,
                    "code": code_snapshot(root), "packages": environment.get("packages"),
                    "os": environment.get("os"), "torch": environment.get("torch", {})}
        # Free GPU memory is transient; it must not invalidate an otherwise identical resume.
        identity["torch"] = dict(identity["torch"], gpus=[{k: v for k, v in gpu.items() if k != "free_bytes"}
                                      for gpu in identity["torch"].get("gpus", [])])
        manifest_id = fingerprint(identity)
        manifest_file = output / "manifest.json"
        if output.exists() and any(output.iterdir()):
            if not resume or not manifest_file.exists():
                raise ValueError("Output directory is nonempty. Use a new directory or explicit --resume.")
            previous = json.loads(manifest_file.read_text(encoding="utf-8"))
            if previous["fingerprint"] != manifest_id:
                raise ValueError("Resume requires identical suite, inputs, source code, model config and software/runtime.")
        output.mkdir(parents=True, exist_ok=True)
        snapshots = {}
        for sample in suite.samples:
            snapshots[sample.id] = {}
            for kind in ("image", "mask"):
                source = getattr(sample, kind)
                if source:
                    target = output / "inputs" / sample.id / f"{kind}{Path(source).suffix}"
                    if manifest_file.exists():
                        if file_hash(target) != hashes[sample.id][kind]:
                            raise ValueError(f"Changed input snapshot: {target}")
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(root / source, target)
                        if file_hash(target) != hashes[sample.id][kind]:
                            raise ValueError("Input changed while freezing the experiment")
                    snapshots[sample.id][kind] = target
        if not manifest_file.exists():
            write_json(manifest_file, {"schema_version": 1, "fingerprint": manifest_id, **identity,
                                     "environment": environment, "planned_cases": [asdict(case) for case in suite.cases()],
                                     "timing_note": "Loading excluded; no dedicated warmup; first-case/shape kernel overhead may remain. Descriptive timings only."})
        samples = {s.id: s for s in suite.samples}
        reports = []
        for case in suite.cases():
            directory = output / "runs" / case.id
            case_id = fingerprint({"manifest": manifest_id, "case": asdict(case)})
            completed = validate_completed(directory, case_id) if resume else None
            if completed is not None:
                logger.info("Reuse completed %s", case.id)
                reports.append(completed)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            try:
                if self.backend is None:
                    self.backend = self._loader(self.config)
                    self.editor, self.reconstructor = EditingPipeline(self.backend), ReconstructionPipeline(self.backend)
                sample, settings = samples[case.sample], case.settings
                with Image.open(snapshots[sample.id]["image"]) as value:
                    image = ImageOps.exif_transpose(value).convert("RGB")
                mask = None
                if sample.mask:
                    with Image.open(snapshots[sample.id]["mask"]) as value:
                        mask = ImageOps.exif_transpose(value).copy()
                torch.manual_seed(settings.seed)
                sync(self.backend.device)
                if self.backend.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with count_unet(self.backend.unet) as counts:
                    if case.operation == "reconstruct":
                        result = self.reconstructor.reconstruct(image, sample.source_prompt, settings.steps, settings.guidance,
                            inversion_guidance_scale=settings.inversion_guidance, negative_prompt=settings.negative_prompt,
                            inversion_refinement_steps=settings.refinement_steps,
                            inversion_refinement_tolerance=settings.refinement_tolerance)
                        images = {"original": result.original, "vae": result.vae_reconstruction, "output": result.ddim_reconstruction}
                        selection = None
                        state_ids = result.sampling.timesteps
                        fidelity = result.fidelity_check
                        method = fidelity.selected_method if fidelity else "plain_ddim"
                    else:
                        options = dict(num_steps=settings.steps, strength=settings.strength, guidance_scale=settings.guidance,
                            inversion_guidance_scale=settings.inversion_guidance, negative_prompt=settings.negative_prompt,
                            inversion_refinement_steps=settings.refinement_steps,
                            inversion_refinement_tolerance=settings.refinement_tolerance)
                        if settings.mask_mode == "global":
                            result = self.editor.edit_global(image, sample.source_prompt, sample.target_prompt, **options)
                        else:
                            result = self.editor.edit_local(image, sample.source_prompt, sample.target_prompt, mask,
                                latent_blending=settings.mask_mode != "pixel_only",
                                preserve_background="pixel" in settings.mask_mode,
                                feather_radius=settings.feather_radius if "feather" in settings.mask_mode else 0, **options)
                        images = {"original": result.original, "vae": result.vae_reconstruction,
                                  "source_control": result.source_reconstruction, "output": result.edited_image}
                        if settings.mask_mode != "global":
                            images.update(uncomposited=result.uncomposited_image, source_uncomposited=result.uncomposited_source)
                        selection, state_ids, fidelity = asdict(result.selection), result.target_sampling.timesteps, None
                        method = "custom_ddim_edit"
                sync(self.backend.device)
                elapsed = time.perf_counter() - started
                masks = {}
                if mask is not None:
                    masks["requested"] = prepare_mask(mask, tuple(result.original.shape[-2:]),
                        (image.height // self.editor.vae.spatial_factor, image.width // self.editor.vae.spatial_factor)).pixel
                    mask.save(directory / "input_mask.png")
                    images["requested_mask"] = masks["requested"].expand(-1, 3, -1, -1)
                if settings.mask_mode != "global":
                    masks.update(effective=result.pixel_mask, latent=result.latent_mask)
                    images["effective_mask"] = result.pixel_mask.expand(-1, 3, -1, -1)
                metrics = calculate_metrics(result.original, images, masks.get("requested"), masks.get("effective"))
                inversion = result.inversion
                report = {"case": asdict(case), "case_fingerprint": case_id, "manifest_fingerprint": manifest_id,
                          "sample": asdict(sample), "inputs": hashes[sample.id], "model_config": self.config.to_dict(),
                          "method": method, "runtime_seconds": elapsed, "budget": counts,
                          "device": self.backend.device, "unet_dtype": str(next(self.backend.unet.parameters()).dtype),
                          "vae_dtype": str(next(self.backend.vae.parameters()).dtype),
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self.backend.device == "cuda" else None,
                          "selection": selection, "sampling_timesteps": state_ids,
                          "inversion_timesteps": inversion.timesteps if inversion else None,
                          "fidelity_check": {"selected_method": fidelity.selected_method,
                                             "refined_latent_mse": fidelity.refined_latent_mse,
                                             "plain_latent_mse": fidelity.plain_latent_mse} if fidelity else None,
                          "refinement_diagnostics": {str(t): asdict(d) for t, d in
                              ((fidelity.candidate_refinement if fidelity else inversion.refinement if inversion else None) or {}).items()}}
                reports.append(save_case(directory, images, metrics, report, masks=masks))
                logger.info("Completed %s: %.3fs, %s", case.id, elapsed, counts)
            except Exception as exc:
                write_json(directory / "error.json", {"case": asdict(case), "error": f"{type(exc).__name__}: {exc}"})
                raise
        return reports
