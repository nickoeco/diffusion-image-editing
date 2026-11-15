"""REFERENCE ONLY: built-in text-to-image generation smoke, not our editor."""
import argparse
from pathlib import Path

from diffusion_editor.config import load_config
from diffusion_editor.models.sd_backend import resolve_runtime
from diffusion_editor.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prompt", default="a red apple on a wooden table")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/generation/reference.png"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    import torch
    from diffusers import StableDiffusionPipeline
    config = load_config(args.config)
    configure_logging(config.log_level)
    device, dtype = resolve_runtime(config, torch)
    kwargs = {"variant": "fp16"} if dtype == torch.float16 else {}
    # Deliberately confined to a reference script. Keep its default safety checker.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model_id, revision=config.revision, torch_dtype=dtype,
        use_safetensors=True, cache_dir=config.cache_dir,
        local_files_only=False if args.allow_download else config.local_files_only, **kwargs).to(device)
    with torch.inference_mode():
        result = pipeline(args.prompt, num_inference_steps=args.steps,
                          generator=torch.Generator(device=device).manual_seed(config.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    print(f"Saved reference image: {args.output}")


if __name__ == "__main__":
    main()
