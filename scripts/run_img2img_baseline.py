"""BASELINE ONLY: Diffusers DDIM Img2Img, never used by the custom editor.

Use the pipeline's supported latent input/output to keep the shared VAE in FP32.
The baseline samples its VAE posterior and adds random noise (no DDIM inversion).
"""
import argparse
from collections.abc import Callable
import json
import logging
from pathlib import Path
import time


def run_baselines(experiment_dir: str | Path, *, root: str | Path = ".", study: str = "strength_global",
                  values: tuple[float, ...] = (0.5, 1.0), resume: bool = False,
                  backend_loader: Callable | None = None) -> list[dict]:
    import torch
    from PIL import Image, ImageOps
    from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
    from diffusion_editor.config import ProjectConfig
    from diffusion_editor.editing.mask import prepare_mask
    from diffusion_editor.evaluation.artifacts import calculate_metrics, count_unet, save_case, sync
    from diffusion_editor.evaluation.experiment_config import file_hash, fingerprint
    from diffusion_editor.evaluation.records import validate_completed
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.models.vae import VAEWrapper
    from diffusion_editor.utils.environment import inspect_environment
    from diffusion_editor.utils.image_utils import image_to_tensor

    output, root = Path(experiment_dir).resolve(), Path(root).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = ProjectConfig(**manifest["model_config"])
    if not config.local_files_only:
        raise ValueError("Baseline requires cached weights, local_files_only=true")
    selected = [case for case in manifest["planned_cases"] if case["study"] == study and case["value"] in values]
    if not selected or set(case["value"] for case in selected) != set(values):
        raise ValueError("Requested baseline values must match the selected completed study")
    custom_reports = []
    for case in selected:
        settings = case["settings"]
        if case["operation"] != "edit" or settings["mask_mode"] != "global" or settings["guidance"] <= 1:
            raise ValueError("This labelled baseline comparison requires global edits with guidance >1")
        expected = fingerprint({"manifest": manifest["fingerprint"], "case": case})
        custom = validate_completed(output / "runs" / case["id"], expected)
        if custom is None:
            raise ValueError(f"Custom case must finish first: {case['id']}")
        if int(settings["steps"] * settings["strength"]) != custom["selection"]["reverse_steps"] or custom["selection"]["reverse_steps"] < 1:
            raise ValueError("Baseline floor strength and custom ceil strength select different reverse budgets; choose matching values")
        for key in ("image", "mask"):
            path = custom["sample"].get(key)
            if path and file_hash(root / path) != custom["inputs"][key]:
                raise ValueError("Baseline inputs differ from the completed custom case")
        custom_reports.append(custom)
    environment = inspect_environment()
    for package in ("torch", "diffusers", "transformers", "numpy", "Pillow"):
        if environment["packages"][package] != manifest["environment"]["packages"][package]:
            raise ValueError(f"Baseline/custom software version differs: {package}")
    pipeline = backend = None
    reports = []
    for custom in custom_reports:
        case, sample = custom["case"], custom["sample"]
        settings = case["settings"]
        directory = output / "baseline" / case["id"]
        identity = {"matched_case_fingerprint": custom["case_fingerprint"], "script_sha256": file_hash(Path(__file__)),
                    "method": "diffusers_ddim_img2img_fp32_vae", "packages": environment["packages"]}
        case_fingerprint = fingerprint(identity)
        if directory.exists() and any(directory.iterdir()):
            if not resume:
                raise ValueError("Baseline output exists; use a new experiment directory or --resume")
            completed = validate_completed(directory, case_fingerprint)
            if completed is not None:
                reports.append(completed)
                continue
        if backend is None:
            backend = (backend_loader or SDBackend.load)(config)
            if backend.device != custom["device"] or str(next(backend.unet.parameters()).dtype) != custom["unet_dtype"]:
                raise ValueError("Baseline/custom device or U-Net precision differs")
            scheduler = DDIMScheduler.from_config(backend.scheduler.config, set_alpha_to_one=True)
            pipeline = StableDiffusionImg2ImgPipeline(vae=backend.vae, tokenizer=backend.tokenizer,
                text_encoder=backend.text_encoder, unet=backend.unet, scheduler=scheduler,
                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
            pipeline.set_progress_bar_config(disable=True)
            vae = VAEWrapper(backend.vae)
        # Use the verified normalized input saved by the matched custom case.
        with Image.open(output / "runs" / case["id"] / "original.png") as value:
            image = ImageOps.exif_transpose(value).convert("RGB")
        original = image_to_tensor(image)
        generator = torch.Generator(device=backend.device).manual_seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        sync(backend.device)
        if backend.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        visited = []
        def record_step(pipe, index, timestep, callback_kwargs):
            visited.append(int(timestep))
            return callback_kwargs
        started = time.perf_counter()
        with torch.inference_mode(), count_unet(backend.unet) as budget:
            posterior = backend.vae.encode(original.to(device=backend.device, dtype=torch.float32) * 2 - 1).latent_dist
            scale = backend.vae.config.scaling_factor
            vae_image = vae.decode_latents(posterior.mode() * scale)
            sampled_latents = posterior.sample(generator=generator) * scale
            final_latents = pipeline(sample["target_prompt"], image=sampled_latents,
                num_inference_steps=settings["steps"], strength=settings["strength"],
                guidance_scale=settings["guidance"], negative_prompt=settings["negative_prompt"],
                eta=0, generator=generator, output_type="latent", callback_on_step_end=record_step).images
            edited = vae.decode_latents(final_latents)
        sync(backend.device)
        elapsed = time.perf_counter() - started
        if visited != custom["sampling_timesteps"][:-1] or budget["unet_calls"] != len(visited):
            raise RuntimeError("Actual baseline reverse grid/budget differs from the matched custom case")
        images = {"original": original, "vae": vae_image, "output": edited}
        masks = {}
        directory.mkdir(parents=True, exist_ok=True)
        if sample.get("mask"):
            with Image.open(output / "runs" / case["id"] / "input_mask.png") as mask:
                masks["requested"] = prepare_mask(mask, tuple(original.shape[-2:]), tuple(final_latents.shape[-2:])).pixel
                mask.save(directory / "input_mask.png")
            images["requested_mask"] = masks["requested"].expand(-1, 3, -1, -1)
        metrics = calculate_metrics(original, images, masks.get("requested"))
        report = {"case_fingerprint": case_fingerprint, "baseline_identity": identity, "matched_case_id": case["id"],
                  "matched_case_fingerprint": custom["case_fingerprint"], "case": case, "sample": sample,
                  "inputs": custom["inputs"], "model_config": config.to_dict(), "environment": environment,
                  "method": "BASELINE: Diffusers DDIM Img2Img with FP32 VAE codec", "runtime_seconds": elapsed,
                  "budget": budget, "sampling_timesteps": visited, "device": backend.device,
                  "unet_dtype": str(next(backend.unet.parameters()).dtype), "vae_dtype": str(next(backend.vae.parameters()).dtype),
                  "final_latent_dtype": str(final_latents.dtype),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None,
                  "differences": ["Random posterior sample and Gaussian noising, no source-conditioned inversion or control branch",
                                  "Same requested steps, reverse timestep IDs, guidance, image size, weights and seed; total U-Net budget differs",
                                  "Built-in scheduler and CFG; baseline latent arithmetic follows pipeline dtype, custom updates use FP32",
                                  "FP32 VAE encode/decode outside pipeline via supported latent input/output; no new weights",
                                  "No safety-checker components loaded for this fixed-input offline evaluation; not a deployment pipeline"]}
        reports.append(save_case(directory, images, metrics, report, masks=masks))
        logging.getLogger(__name__).info("BASELINE completed %s: %.3fs, %s", case["id"], elapsed, budget)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--study", default="strength_global")
    parser.add_argument("--values", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from diffusion_editor.utils.logging import configure_logging
    configure_logging("INFO")
    reports = run_baselines(args.experiment_dir, root=args.root, study=args.study, values=args.values, resume=args.resume)
    from diffusion_editor.evaluation.report import summarize
    summarize(args.experiment_dir)
    print(f"Completed {len(reports)} labelled baseline cases")


if __name__ == "__main__":
    main()
