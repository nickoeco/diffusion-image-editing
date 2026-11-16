"""Cached-model global or mask-guided editing with a matched source-prompt control."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from diffusion_editor.config import load_config
from diffusion_editor.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", type=Path, default=Path("assets/examples/astronaut.png"))
    parser.add_argument("--source-prompt", default="a portrait of an astronaut wearing an orange spacesuit")
    parser.add_argument("--target-prompt", default="a portrait of an astronaut wearing a blue spacesuit")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--mask", type=Path, help="Aligned single-channel mask image: white edits, black preserves")
    parser.add_argument("--soft-mask", action="store_true")
    parser.add_argument("--feather-radius", type=float, default=0)
    parser.add_argument("--no-latent-blending", action="store_true")
    parser.add_argument("--no-pixel-composite", action="store_true")
    parser.add_argument("--refinement-steps", type=int, default=0, help="Optional source inversion residual evaluations")
    parser.add_argument("--refinement-tolerance", type=float, default=1e-6)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/editing/phase4"))
    args = parser.parse_args()
    if not args.mask and (args.soft_mask or args.feather_radius or args.no_latent_blending or args.no_pixel_composite):
        parser.error("mask options require --mask")
    import torch
    from PIL import Image
    from diffusion_editor.evaluation.metrics import reconstruction_metrics, background_metrics
    from diffusion_editor.editing.mask import prepare_mask
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
    from diffusion_editor.utils.environment import inspect_environment
    from diffusion_editor.utils.image_utils import tensor_to_images
    from diffusion_editor.utils.visualization import save_image_grid

    config = load_config(args.config)
    if not config.local_files_only:
        parser.error("editing requires cached weights (local_files_only=true)")
    configure_logging(config.log_level)
    torch.manual_seed(config.seed)
    backend = SDBackend.load(config)
    editor = EditingPipeline(backend)
    if backend.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with Image.open(args.image) as image:
        started = time.perf_counter()
        options = dict(strength=args.strength, guidance_scale=args.guidance_scale, num_steps=args.steps,
                       inversion_guidance_scale=args.inversion_guidance_scale,
                       negative_prompt=args.negative_prompt, return_trajectory=args.save_trajectories,
                       inversion_refinement_steps=args.refinement_steps,
                       inversion_refinement_tolerance=args.refinement_tolerance)
        if args.mask:
            with Image.open(args.mask) as mask:
                result = editor.edit_local(image, args.source_prompt, args.target_prompt, mask,
                                           binary_mask=not args.soft_mask, feather_radius=args.feather_radius,
                                           latent_blending=not args.no_latent_blending,
                                           preserve_background=not args.no_pixel_composite, **options)
                requested_mask = prepare_mask(mask, tuple(result.original.shape[-2:]),
                                              tuple(result.latent_mask.shape[-2:]), binary=not args.soft_mask).pixel
        else:
            result = editor.edit_global(image, args.source_prompt, args.target_prompt, **options)
    if backend.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    background = None
    if args.mask:
        background = {"uncomposited_effective_mask": background_metrics(result.original, result.uncomposited_image, result.pixel_mask),
                      "final_effective_mask": background_metrics(result.original, result.edited_image, result.pixel_mask),
                      "final_requested_mask": background_metrics(result.original, result.edited_image, requested_mask)}
        background = {stage: {key: None if isinstance(value, float) and not math.isfinite(value) else value
                               for key, value in values.items()} for stage, values in background.items()}
    metrics = {"vae_vs_original": reconstruction_metrics(result.original, result.vae_reconstruction),
               "source_vs_original": reconstruction_metrics(result.original, result.source_reconstruction),
               "edited_vs_original": reconstruction_metrics(result.original, result.edited_image),
               "edited_vs_source_control": reconstruction_metrics(result.source_reconstruction, result.edited_image)}
    metrics = {stage: {key: value if math.isfinite(value) else None for key, value in values.items()}
               for stage, values in metrics.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [result.original, result.vae_reconstruction, result.source_reconstruction, result.edited_image]
    for name, image in zip(("original", "vae_reconstruction", "source_reconstruction", "edited"), images):
        tensor_to_images(image)[0].save(args.output_dir / f"{name}.png")
    save_image_grid(images, ["Original", "VAE only", "Source-prompt control", "Target-prompt edit"],
                    args.output_dir / "comparison.png", columns=4)
    if args.mask:
        for name, tensor in (("edited_uncomposited", result.uncomposited_image),
                             ("source_uncomposited", result.uncomposited_source)):
            tensor_to_images(tensor)[0].save(args.output_dir / f"{name}.png")
        effective = result.pixel_mask.detach().cpu()
        Image.fromarray((effective[0, 0] * 255).round().byte().numpy()).save(args.output_dir / "effective_mask.png")
        torch.save({"requested": requested_mask.cpu(), "pixel": effective,
                    "latent": result.latent_mask.cpu()}, args.output_dir / "mask_tensors.pt")
        save_image_grid([result.original, effective.expand(-1, 3, -1, -1), result.uncomposited_image, result.edited_image],
                        ["Original", "Effective edit mask", "Before pixel compositing", "Final output"],
                        args.output_dir / "preservation_comparison.png", columns=4)
    if args.save_trajectories:
        torch.save({"inversion": dict(result.inversion.trajectory) if result.inversion else None,
                    "source": dict(result.source_sampling.trajectory), "target": dict(result.target_sampling.trajectory)},
                   args.output_dir / "trajectories.pt")
    report = {"config": config.to_dict(), "source_image": str(args.image),
              "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "source_prompt": args.source_prompt, "target_prompt": args.target_prompt,
              "mode": "local" if args.mask else "global",
              "mask_file": str(args.mask) if args.mask else None,
              "mask_sha256": hashlib.sha256(args.mask.read_bytes()).hexdigest() if args.mask else None,
              "mask_settings": {"binary": not args.soft_mask, "feather_radius_pixels": args.feather_radius,
                                "latent_blending": not args.no_latent_blending,
                                "pixel_compositing": not args.no_pixel_composite,
                                "blend_timing": "after update, source at updated timestep including clean"} if args.mask else None,
              "background_metrics": background,
              "background_metric_note": "Weights=1-M; zero-mask MAE uses only M==0. Empty regions are undefined; null PSNR with positive weight denotes exact preservation." if args.mask else None,
              "negative_prompt": args.negative_prompt, "num_steps": args.steps,
              "inversion_guidance_scale": args.inversion_guidance_scale, "guidance_scale": args.guidance_scale,
              "inversion_method": "residual_refined_ddim" if args.refinement_steps else "plain_ddim",
              "refinement_steps": args.refinement_steps, "refinement_tolerance": args.refinement_tolerance,
              "refinement_diagnostics": {str(t): asdict(d) for t, d in result.inversion.refinement.items()}
                  if result.inversion and result.inversion.refinement else None,
              "selection": asdict(result.selection) | {"noise_std": result.selection.noise_std},
              "inversion_steps_executed": args.steps if result.inversion else 0,
              "inversion_timesteps": result.inversion.timesteps if result.inversion else None,
              "source_timesteps": result.source_sampling.timesteps, "target_timesteps": result.target_sampling.timesteps,
              "clean_boundary": {"id": -1, "alpha": 1.0}, "image_shape": list(result.edited_image.shape),
              "preservation_metrics": metrics, "metrics_note": "PSNR/SSIM do not measure target-prompt alignment; null PSNR means exact match",
              "metric_settings": {"data_range": 1, "color": "RGB", "ssim_window": 7,
                                  "ssim_gaussian_weights": False, "ssim_sample_covariance": True},
              "runtime_seconds_including_source_control": elapsed,
              "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None,
              "environment": inspect_environment()}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "psnr_db", "ssim"])
        writer.writeheader()
        writer.writerows({"comparison": key, **values} for key, values in metrics.items())
    print(json.dumps({key: value for key, value in report.items() if key != "environment"}, indent=2))


if __name__ == "__main__":
    main()
