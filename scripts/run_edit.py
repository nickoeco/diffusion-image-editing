"""Cached-model global text editing with a matched same-prompt control."""
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
    parser.add_argument("--refinement-steps", type=int, default=0, help="Optional source inversion residual evaluations")
    parser.add_argument("--refinement-tolerance", type=float, default=1e-6)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/editing/phase4"))
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusion_editor.evaluation.metrics import reconstruction_metrics
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.editing_pipeline import EditingPipeline
    from diffusion_editor.utils.environment import inspect_environment
    from diffusion_editor.utils.image_utils import tensor_to_images
    from diffusion_editor.utils.visualization import save_image_grid

    config = load_config(args.config)
    if not config.local_files_only:
        parser.error("global editing requires cached weights (local_files_only=true)")
    configure_logging(config.log_level)
    torch.manual_seed(config.seed)
    backend = SDBackend.load(config)
    editor = EditingPipeline(backend)
    if backend.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with Image.open(args.image) as image:
        started = time.perf_counter()
        result = editor.edit_global(image, args.source_prompt, args.target_prompt, args.strength,
                                    args.guidance_scale, args.steps,
                                    inversion_guidance_scale=args.inversion_guidance_scale,
                                    negative_prompt=args.negative_prompt, return_trajectory=args.save_trajectories,
                                    inversion_refinement_steps=args.refinement_steps,
                                    inversion_refinement_tolerance=args.refinement_tolerance)
    if backend.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
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
    if args.save_trajectories:
        torch.save({"inversion": dict(result.inversion.trajectory) if result.inversion else None,
                    "source": dict(result.source_sampling.trajectory), "target": dict(result.target_sampling.trajectory)},
                   args.output_dir / "trajectories.pt")
    report = {"config": config.to_dict(), "source_image": str(args.image),
              "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "source_prompt": args.source_prompt, "target_prompt": args.target_prompt,
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
