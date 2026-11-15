"""Opt-in, cached-weight DDIM inversion smoke test; no reverse reconstruction yet."""
import argparse
import json
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inversion/phase2"))
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusion_editor.diffusion.schedule import DDIMSchedule
    from diffusion_editor.diffusion.inversion import DDIMInverter
    from diffusion_editor.models.noise_predictor import UNetNoisePredictor
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.models.vae import VAEWrapper
    from diffusion_editor.models.text_encoder import TextEncoderWrapper
    from diffusion_editor.utils.environment import inspect_environment

    config = load_config(args.config)
    if not config.local_files_only:
        parser.error("inversion smoke requires local_files_only=true; cache weights first")
    configure_logging(config.log_level)
    backend = SDBackend.load(config)
    schedule = DDIMSchedule.from_scheduler(backend.scheduler, args.steps)
    with Image.open(args.image) as image:
        clean = VAEWrapper(backend.vae).encode_image(image)
    embeddings = TextEncoderWrapper(backend.tokenizer, backend.text_encoder).encode_prompt(args.source_prompt)
    if backend.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = DDIMInverter(schedule, UNetNoisePredictor(backend.unet)).invert(clean, embeddings)
    if backend.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(result.trajectory), args.output_dir / "trajectory.pt")
    report = {"config": config.to_dict(), "source_image": str(args.image), "source_prompt": args.source_prompt,
              "steps": args.steps, "guidance_scale": result.guidance_scale,
              "timesteps": result.timesteps, "prediction_timesteps": result.prediction_timesteps,
              "alphas_cumprod_at_levels": {str(t): schedule.alpha(t).item() for t in result.timesteps},
              "clean_boundary": {"id": -1, "alpha": 1.0, "set_alpha_to_one_override": True},
              "latent_shape": list(result.final_latent.shape), "trajectory_dtype": str(result.final_latent.dtype),
              "trajectory_storage": "cpu", "runtime_seconds_inversion_only": elapsed,
              "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None,
              "finite": all(bool(torch.isfinite(z).all()) for z in result.trajectory.values()),
              "environment": inspect_environment()}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
