"""Cached-model same-prompt reconstruction, metrics and optional latent previews."""
import argparse
import csv
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
    parser.add_argument("--image", type=Path, default=Path("outputs/reconstruction/phase1/original.png"))
    parser.add_argument("--source-prompt", default="a red apple on a colorful background")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction/phase3"))
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusion_editor.evaluation.metrics import reconstruction_metrics
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
    from diffusion_editor.utils.environment import inspect_environment
    from diffusion_editor.utils.image_utils import tensor_to_images
    from diffusion_editor.utils.visualization import save_image_grid

    config = load_config(args.config)
    if not config.local_files_only:
        parser.error("reconstruction smoke requires cached weights (local_files_only=true)")
    configure_logging(config.log_level)
    torch.manual_seed(config.seed)
    backend = SDBackend.load(config)
    pipeline = ReconstructionPipeline(backend)
    with Image.open(args.image) as image:
        if backend.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = pipeline.reconstruct(image, args.source_prompt, args.steps, args.guidance_scale,
                                      inversion_guidance_scale=args.inversion_guidance_scale,
                                      negative_prompt=args.negative_prompt,
                                      return_trajectory=args.save_intermediates)
    if backend.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {"vae_only": reconstruction_metrics(result.original, result.vae_reconstruction),
               "ddim_reconstruction": reconstruction_metrics(result.original, result.ddim_reconstruction),
               "ddim_vs_vae": reconstruction_metrics(result.vae_reconstruction, result.ddim_reconstruction)}
    # Strict JSON: exact-match PSNR (+inf) is encoded as null, documented in report.
    metrics = {stage: {k: value if math.isfinite(value) else None for k, value in values.items()}
               for stage, values in metrics.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [result.original, result.vae_reconstruction, result.ddim_reconstruction]
    names = ["original", "vae_reconstruction", "ddim_reconstruction"]
    for image, name in zip(images, names):
        tensor_to_images(image)[0].save(args.output_dir / f"{name}.png")
    save_image_grid(images, ["Original", "VAE only", "DDIM reconstruction"], args.output_dir / "comparison.png")
    if args.save_intermediates:
        previews, labels = [], []
        for name, levels, trajectory in (("Inversion", result.inversion.timesteps, result.inversion.trajectory),
                                         ("Sampling", result.sampling.timesteps, result.sampling.trajectory)):
            for index in (0, len(levels) // 2, len(levels) - 1):
                t = levels[index]
                previews.append(pipeline.vae.decode_latents(trajectory[t]))
                labels.append(f"{name}: decoded latent t={t}")
        save_image_grid(previews, labels, args.output_dir / "latent_previews.png")
        torch.save({"inversion": dict(result.inversion.trajectory), "sampling": dict(result.sampling.trajectory)},
                   args.output_dir / "trajectories.pt")
    report = {"config": config.to_dict(), "source_image": str(args.image),
              "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "source_prompt": args.source_prompt, "negative_prompt": args.negative_prompt,
              "num_steps": args.steps, "guidance_scale": args.guidance_scale,
              "inversion_guidance_scale": args.inversion_guidance_scale,
              "inversion_timesteps": result.inversion.timesteps, "reverse_timesteps": result.sampling.timesteps,
              "clean_boundary": {"id": -1, "alpha": 1.0}, "metrics": metrics,
              "metric_settings": {"data_range": 1, "color": "RGB", "ssim_window": 7,
                                  "ssim_gaussian_weights": False, "ssim_sample_covariance": True,
                                  "psnr_null_means_exact_match": True},
              "image_shape": list(result.original.shape), "latent_dtype": str(result.sampling.final_latent.dtype),
              "runtime_seconds_reconstruction_only": elapsed,
              "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None,
              "environment": inspect_environment()}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "psnr_db", "ssim"])
        writer.writeheader()
        writer.writerows({"comparison": key, **values} for key, values in metrics.items())
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
